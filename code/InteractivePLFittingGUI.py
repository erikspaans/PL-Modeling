"""
PL Modeling Program
-> Python file 1/3: InteractivePLFittingGUI.py (implements the GUI)
Python file 2/3: PLModeling.py (implements the PL emission models)
Python file 3/3: InterferenceFunction.py (implements the interference function models)

Author: Erik Spaans
Date: March 2021
"""
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
import scipy.constants as const
from InterferenceFunction import IF
from PLModeling import PLModel
from pathlib import Path
from itertools import compress
from os import path
import sys
import configparser
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk

# Define constant
h_eV = const.value('Planck constant in eV/Hz')

# Set proper path for files when executable is run
path_to_file = getattr(sys, '_MEIPASS', path.abspath(path.dirname(__file__)))


class Parameter:
    """Class that defines a model parameter."""
    def __init__(self, value, bounds, s_label, s_unit, s_scale, s_prec):
        self.value = value
        self.bounds = bounds
        self.s_label = s_label
        self.s_unit = s_unit
        self.s_scale = s_scale
        self.s_prec = s_prec


class Model:
    """Class that defines a model."""
    def __init__(self, model_class, parameters, x_var, label, active_model):
        self.model_class = model_class
        self.parameters = parameters
        self.xunit = x_var
        self.label = label
        self.active_model = active_model


class Slider(ttk.Scale):
    """
    ttk.Scale sublass that limits the precision of values. Taken from:
    'https://stackoverflow.com/questions/54186639/tkinter-control-ttk-scales-
    increment-as-with-tk-scale-and-a-tk-doublevar'
    """

    def __init__(self, *args, **kwargs):
        self.precision = kwargs.pop('precision')  # Remove non-std kwarg.
        self.chain = kwargs.pop('command', lambda *a: None)  # Save if present.
        super(Slider, self).__init__(*args, command=self._value_changed, **kwargs)

    def _value_changed(self, newvalue):
        if self.precision == 0:
            newvalue = int(round(float(newvalue), 0))
        else:
            newvalue = round(float(newvalue), self.precision)
        self.winfo_toplevel().globalsetvar(self.cget('variable'), (newvalue))
        self.chain(newvalue)  # Call user specified function.


class NotebookTabs:
    """Class that manages the tabs in the PL modeling program."""
    ebox_width = 5
    def __init__(self, window):
        self.window = window
        self.font_title = tkFont.Font(weight='bold', size=18)
        self.font_subtitle = tkFont.Font(weight='bold', size=16)
        self.font_legend = tkFont.Font(weight='bold', size=11)
        self.font_label = tkFont.Font(size=11)

        # Config file
        self.opt_data = None
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(path.join(path_to_file, 'files', 'config_default.ini'))
        self.c_data = self.fetchConfigData()

        # Create tabs
        self.tab_control = ttk.Notebook(self.window)
        # Main tab
        self.models = self.createIFPLModels(np.array(1e-6), np.array(1))
        self.objects_main = {'IF': {}, 'PL': {}}
        self.tree = {'frame': None, 'tree': None}
        self.main = self.createMainTab()
        self.tabs = {'main': self.main['frame'], 'data': {}}

        # Update IF tab when selected
        self.tab_control.bind("<<NotebookTabChanged>>", self.updateSummary)

        # Choose model
        self.updateModel('IF')
        self.updateModel('PL')

        # Place tabs
        self.tab_control.add(self.main['frame'], text='Main')
        self.tab_control.pack(expand=1, fill='both')

    def updateSummary(self, event):
        sel_tab = event.widget.select()
        tab_name = event.widget.tab(sel_tab, "text")
        if tab_name == 'IF analysis':
            self.tabs['summary']['GUI'].updateGraph()

    def createMainTab(self):
        # Create tab
        tab_main = ttk.Frame(self.tab_control)
        tab_main.rowconfigure(2, weight=1)
        tab_main.columnconfigure(1, weight=1)
        # Title
        frm_title = ttk.Frame(tab_main)
        (tk.Label(master=frm_title, text='PL Modeling Program', font=self.font_title)
         .grid(row=0, column=1, columnspan=4, sticky='news', pady=5))
        # Logo
        image = Image.open(path.join(path_to_file, 'files', "logo.ico")).resize((80, 80))
        photo = ImageTk.PhotoImage(image)
        lbl = tk.Label(master=frm_title, image=photo)
        lbl.image = photo
        lbl.grid(row=0, column=0, rowspan=3, padx=10, pady=10, sticky='w')
        # Open file button
        btn_open = tk.Button(master=frm_title, text='Open data', font=self.font_label, command=self.openFile)
        btn_open.grid(row=1, column=1, columnspan=1, pady=5, padx=5, sticky='w')
        # Unit
        frm_import_unit = ttk.Frame(frm_title)
        (tk.Label(master=frm_import_unit, text='Unit:', font=self.font_legend)
         .grid(row=0, column=0, sticky='w'))
        c_box_import = ttk.Combobox(frm_import_unit, values=['nm', 'μm', 'eV'], state='readonly', width=5)
        c_box_import.current(0)
        c_box_import.grid(row=0, column=1, sticky='w')
        frm_import_unit.grid(row=2, column=1, sticky='w', pady=5, padx=5)
        # Close all tabs
        btn_close_all = tk.Button(master=frm_title, text='Close tabs', font=self.font_label, command=self.closeTabs)
        btn_close_all.grid(row=1, column=2, sticky='w', pady=2, padx=5)
        # Exit program
        btn_exit = tk.Button(master=frm_title, text='Exit program', font=self.font_label, command=self.exitProgram)
        btn_exit.grid(row=2, column=2, sticky='w', pady=2, padx=5)
        # Extra space
        frm_title.columnconfigure(1, minsize=150)
        frm_title.columnconfigure(2, minsize=150)
        frm_title.columnconfigure(3, minsize=150)
        # Config file
        frm_config = ttk.Frame(frm_title)
        btn_config = tk.Button(master=frm_config, text='Load config file', font=self.font_label, command=self.loadConfig)
        btn_config.grid(row=0, column=0, pady=5)
        # Labels
        (tk.Label(master=frm_config, text='File:', font=self.font_legend)
         .grid(row=0, column=1, sticky='w'))
        file_var = tk.StringVar()
        file_var.set('default')
        (tk.Label(master=frm_config, textvariable=file_var, font=self.font_label)
         .grid(row=0, column=2, sticky='w'))
        frm_config.grid(row=1, column=3, sticky='w', padx=5, pady=5)
        # Open IF tab
        btn_IFtab = tk.Button(master=frm_title, text='Open IF analysis tab', font=self.font_label, command=self.openIFTab)
        btn_IFtab.grid(row=2, column=3, sticky='w', pady=2, padx=5)
        # Fit all
        frm_fit_params = ttk.Frame(tab_main)
        frm_lbl_fit = ttk.Frame(frm_fit_params)
        (tk.Label(master=frm_lbl_fit, text='Fit all data', font=self.font_subtitle)
         .grid(row=0, column=0, padx=5))
        image = Image.open(path.join(path_to_file, 'files', "help.png")).resize((20,20))
        help_image = ImageTk.PhotoImage(image)
        btn_info = tk.Button(master=frm_lbl_fit, image=help_image, font=self.font_label, command=self.showInfo)
        btn_info.image = help_image
        btn_info.grid(row=0, column=1, pady=5, padx=5)
        # Fit unit
        frm_fit_unit = ttk.Frame(frm_fit_params)
        (tk.Label(master=frm_fit_unit, text='x-axis variable:', font=self.font_legend)
         .grid(row=0, column=0, sticky='w'))
        c_box_fit = ttk.Combobox(frm_fit_unit, values=['wavelength', 'energy'], state='readonly', width=12)
        c_box_fit.bind('<<ComboboxSelected>>', lambda x:self.createTree())
        c_box_fit.current(0)
        c_box_fit.grid(row=0, column=1, sticky='w')
        frm_fit_unit.grid(row=1, column=0, columnspan=2, sticky='ns', pady=5, padx=5)
        frm_lbl_fit.grid(row=0, column=0, columnspan=2, sticky='w')
        # Buttons
        btn_fitPL_all = tk.Button(master=frm_fit_params, text='Fit PL', font=self.font_label, command=self.fitPL_All)
        btn_fitPL_all.grid(row=2, column=0, sticky='news', pady=5, padx=5)
        btn_fitIFPL_all = tk.Button(master=frm_fit_params, text='Fit IF & PL', font=self.font_label, command=self.fitIF_PL_All)
        btn_fitIFPL_all.grid(row=2, column=1, sticky='news', pady=5, padx=5)
        # Comboboxes
        tk.Label(master=frm_fit_params, text='Fit parameters', font=self.font_legend).grid(row=3, column=0, columnspan=1, sticky='news')
        var_select_all = tk.BooleanVar()
        ttk.Checkbutton(master=frm_fit_params, variable=var_select_all, command=self.selectAllFit, text='Select all').grid(row=3, column=1)
        self.objects_main['var_sel_all'] = var_select_all
        self.objects_main['IF']['cbox'] = self.createCombobox(frm_fit_params, 'IF', initial='off')
        self.objects_main['PL']['cbox'] = self.createCombobox(frm_fit_params, 'PL', initial='BGF')
        tk.Label(master=frm_fit_params, text='IF model', font=self.font_label).grid(row=4, column=0, sticky='news')
        tk.Label(master=frm_fit_params, text='PL model', font=self.font_label).grid(row=4, column=1, sticky='news')
        self.objects_main['IF']['cbox'].grid(row=5, column=0, sticky='w', padx=5)
        self.objects_main['PL']['cbox'].grid(row=5, column=1, sticky='w', padx=5)
        # Fixed param
        frm_fixed_IF = ttk.Frame(frm_fit_params)
        frm_fixed_PL = ttk.Frame(frm_fit_params)
        self.createFitFixed(frm_fixed_IF, 'IF')
        self.createFitFixed(frm_fixed_PL, 'PL')
        frm_fixed_IF.grid(row=6, column=0, sticky='n')
        frm_fixed_PL.grid(row=6, column=1, sticky='n')
        # Tree with fit details
        self.tree['frame'] = ttk.Frame(tab_main)
        self.tree['frame'].columnconfigure(0, minsize=150, weight=1)
        self.tree['frame'].columnconfigure(1, minsize=150)
        self.tree['frame'].rowconfigure(1, weight=1)
        (tk.Label(master=self.tree['frame'], text='Fit results', font=self.font_subtitle)
         .grid(row=0, column=0, columnspan=2, sticky='w'))
        self.createTree()
        # Save fit results button
        btn_save = tk.Button(master=self.tree['frame'], text='Save fit results', font=self.font_label, command=self.saveFitsData)
        btn_save.grid(row=2, column=0, sticky='w', pady=5)
        btn_export_plots = tk.Button(master=self.tree['frame'], text='Export all plot data', font=self.font_label, command=self.exportPlotsData)
        btn_export_plots.grid(row=2, column=1, sticky='w', pady=5)

        # Place frames
        frm_title.grid(row=0, column=0, columnspan=2, sticky='ns')
        ttk.Separator(master=tab_main, orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=2, pady=10, sticky='news')
        frm_fit_params.grid(row=2, column=0, sticky='n')
        self.tree['frame'].grid(row=2, column=1, sticky='news')
        return {'frame': tab_main, 'btn_open': btn_open, 'file_var': file_var,
                'btn_config': btn_config, 'btn_fitPL': btn_fitPL_all,
                'btn_fitIF_PL': btn_fitIFPL_all, 'c_box_import': c_box_import,
                'c_box_fit': c_box_fit}

    def showInfo(self):
        message = ('Model info:\n'
                   'IF model: Interference Function model [1].\n'
                   'PL model: Photoluminescence model.\n'
                   '   BGF: band gap fluctuations [2].\n'
                   '   EF: electrostatic fluctuations [3].\n'
                   '   UPF: unified potential fluctuations (BGF + EF) [4].\n\n'
                   'References:\n[1] {}\n[2] {}\n[3] {}\n[4] {}').format('Journal of Applied Physics 118, 035307 (2015); doi: 10.1063/1.4926857',
                   'Journal of Applied Physics 101, 113519 (2007); doi: 10.1063/1.2721768',
                   'Journal of Applied Physics 116, 173504 (2014); doi: 10.1063/1.4898346',
                   'Check repository README.')
        tk.messagebox.showinfo('About', message)

    def exportPlotsData(self):
        filepath = asksaveasfilename(defaultextension='csv', filetypes=[("CSV", "*.csv"), ("All Files", "*.*")])
        if not filepath:
            return
        graph_data = {}
        for idx, tab in enumerate(self.tabs['data']):
            x_var = self.main['c_box_fit'].get()
            if x_var == 'wavelength':
                self.tabs['data'][tab]['GUI'].objects['options']['r_btn_wave'].invoke()
            elif x_var == 'energy':
                self.tabs['data'][tab]['GUI'].objects['options']['r_btn_energy'].invoke()
            if idx == 0:
                graph_data = {**graph_data, **{self.tabs['data'][tab]['GUI'].axes[1].get_xlabel(): self.tabs['data'][tab]['GUI'].graphs[0].get_xdata()}}
            graph_data = {**graph_data, **self.tabs['data'][tab]['GUI'].getPlotData()}
        graph_data = pd.DataFrame(graph_data)
        try:
            graph_data.to_csv(filepath, index=False, encoding="utf-8-sig")
            message = 'Plot data saved to {}.'.format(filepath)
            tk.messagebox.showinfo('Plot data saved successfully', message)
        except:
            message = 'Not able to save data to {}. The file might be open.'.format(filepath)
            tk.messagebox.showerror('Save error', message)

    def loadConfig(self):
        filepath = askopenfilename()
        if not filepath:
            return
        filename = path.basename(filepath)

        config_sec = {'IF': ['nk_dir', 'thickness', 'wave_laser', 'R_rms',
                             'k', 'delta', 'mu', 'sigma'],
                      'BGF': ['E_g', 'beta', 'sigma_g', 'T'],
                      'EF': ['E_g', 'theta', 'gamma', 'delta_mu', 'T', 'a_0d'],
                      'UPF': ['E_g', 'beta', 'sigma_g', 'theta', 'gamma', 'T',
                              'a_0d']}

        # Update config file
        try:
            self.config = configparser.ConfigParser()
            self.config.optionxform = str
            self.config.read(filepath)
            if not all([self.config.has_option(sec, key) for sec in config_sec for key in config_sec[sec]]):
                raise KeyError()
            self.c_data = self.fetchConfigData()
            self.main['file_var'].set(filename)
        except KeyError:
            message = 'Error in configuration file. Please check that the required sections and values are present.'
            tk.messagebox.showerror('File format error', message)
        except ValueError:
            message = 'Error in configuration file. Please check that parameter values are within bounds and try again.'
            tk.messagebox.showerror('Value error', message)
        except:
            message = 'Error in configuration file. Please check file format and try again.'
            tk.messagebox.showerror('File format error', message)

    def fetchConfigData(self):
        config = {}
        for sec in self.config.items():
            params = {}
            for it in self.config.items(sec[0]):
                if it[0] == 'nk_dir':
                    if it[1] == 'default':
                        pass
                    else:
                        nk_dir = Path(it[1])
                        nk_air = pd.read_csv(path.join(nk_dir, 'files', "nk_air.csv"))
                        nk_CIGS = pd.read_csv(path.join(nk_dir, 'files', "nk_CIGS.csv"))
                        nk_Mo = pd.read_csv(path.join(nk_dir, 'files', "nk_Mo.csv"))
                        self.opt_data = {'air': nk_air, 'CIGS': nk_CIGS,
                                         'Mo': nk_Mo}
                else:
                    data = json.loads(it[1])
                    if type(data) is list:
                        if not(data[0] <= data[1] and data[1] <= data[2]):
                            raise ValueError
                    params[it[0]] = json.loads(it[1])
            config[sec[0]] = params
        return config


    def closeTabs(self):
        # close tabs
        for tab in range(1, self.tab_control.index('end')):
            self.tab_control.forget(1)
        # delete data
        try:
            self.tabs['data'] = {}
            del self.tabs['summary']
        except:
            pass
        self.window.title('PL Modeling Program')
        self.createTree()
        self.main['btn_open'].config(state='normal')
        self.main['btn_config'].config(state='normal')

    def exitProgram(self):
        message = 'Are you sure you want to exit the program?'
        answer = tk.messagebox.askyesnocancel('Exit program', message)
        if answer is True:
            self.window.destroy()
        else:
            return

    def openIFTab(self):
        if 'summary' in self.tabs:
            self.tab_control.add(self.tabs['summary']['frame'], text='IF analysis')
            self.tab_control.pack(expand=1, fill='both')

    def saveFitsData(self):
        output_dict = {}
        for idx, col in enumerate([self.tree['tree'].heading('#0')['text']]+list(self.tree['tree']['columns'])):
            output_list = []
            for child in self.tree['tree'].get_children():
                if idx == 0:
                    output_list.append(self.tree['tree'].item(child, option='text'))
                else:
                    output_list.append(self.tree['tree'].item(child, option='values')[idx-1])
            output_dict[col] = output_list
        output = pd.DataFrame(output_dict)
        if output.empty:
            message = 'Please fit the data with a model before saving.'
            tk.messagebox.showwarning('No fit data to save', message)
            return
        else:
            filepath = asksaveasfilename(defaultextension='csv', filetypes=[("CSV", "*.csv"), ("All Files", "*.*")])
            if not filepath:
                return
            try:
                output.to_csv(filepath, index=False, encoding="utf-8-sig")
                message = 'Fit data saved to {}.'.format(filepath)
                tk.messagebox.showinfo('Fit data saved successfully', message)
            except:
                message = 'Not able to save data to {}. The file might be open.'.format(filepath)
                tk.messagebox.showerror('Save error', message)

    def createTree(self):
        if self.tree['tree'] is not None:
            self.tree['tree'].destroy()
        frame_all = ttk.Frame(self.tree['frame'])
        frame_all.columnconfigure(0, minsize=200, weight=1)
        frame_all.rowconfigure(0, minsize=200, weight=1)
        self.tree['tree'] = ttk.Treeview(frame_all)
        width = 80
        min_width = 80
        self.tree['tree'].column('#0', width=width, minwidth=min_width)
        self.tree['tree'].heading('#0', text='Data')
        labels = []
        for label in list(self.models.keys()):
            for param in self.models[label].parameters[self.models[label].active_model]:
                labels.append('{} [{}]'.format(param.s_label, param.s_unit))
        labels.append('Model error')
        self.tree['tree']["columns"] = labels
        for name in labels:
            self.tree['tree'].column(name, width=width, minwidth=min_width)
            self.tree['tree'].heading(name, text=name)

        self.tree['tree'].grid(row=0, column=0, sticky='news')
        # Add scrollbars
        vsb = ttk.Scrollbar(frame_all, orient='vertical', command=self.tree['tree'].yview)
        vsb.grid(row=0, column=1, sticky='ns')
        self.tree['tree'].configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(frame_all, orient='horizontal', command=self.tree['tree'].xview)
        hsb.grid(row=1, column=0, sticky='we')
        self.tree['tree'].configure(xscrollcommand=hsb.set)

        frame_all.grid(row=1, column=0, columnspan=2, sticky='news')


    def selectAllFit(self):
        if self.objects_main['var_sel_all'].get() == 1:
            set_val = 1
        else:
            set_val = 0
        for label in list(self.models.keys()):
            for var in self.objects_main[label][self.models[label].active_model]['var_cbtn']:
                var.set(set_val)
            self.checkFitFixed(label)

    def createFitFixed(self, frame, label):
        # Define parameter widgets
        model = self.models[label]
        strvar_mod = tk.StringVar()
        strvar_mod.set(list(model.parameters.keys())[self.objects_main[label]['cbox'].current()])
        self.objects_main[label]['active'] = strvar_mod
        for idx, mod in enumerate(model.parameters.keys()):
            # Initialise containers
            objects = {}
            cbtn_param = []
            var_cbtn_param = []
            lbl_param = []
            ebox_param = []
            var_ebox_param = []
            unit_param = []
            # Checkbuttons, labels, entryboxes and units
            for idx, param in enumerate(model.parameters[mod]):
                # Checkbutton
                intvar_check = tk.BooleanVar()
                intvar_check.set(True)
                cbtn = ttk.Checkbutton(master=frame, variable=intvar_check, command=lambda:self.checkFitFixed(label))
                cbtn_param.append(cbtn)
                var_cbtn_param.append(intvar_check)
                # Label
                lbl = tk.Label(master=frame, text=param.s_label+': ', font=self.font_label)
                lbl_param.append(lbl)
                # Entrybox
                strvar_ebox = tk.StringVar()
                ebox = tk.Entry(master=frame, textvariable=strvar_ebox, width=self.ebox_width)
                ebox_param.append(ebox)
                var_ebox_param.append(strvar_ebox)
                # Unit
                unit = tk.Label(master=frame, text=param.s_unit, font=self.font_label)
                unit_param.append(unit)
            objects['cbtn'] = cbtn_param
            objects['var_cbtn'] = var_cbtn_param
            objects['lbl'] = lbl_param
            objects['ebox'] = ebox_param
            objects['var_ebox'] = var_ebox_param
            objects['unit'] = unit_param
            self.objects_main[label][mod] = objects

    def updateModel(self, label):
        # Update display
        # Remove present widgets
        for idx in range(len(self.models[label].parameters[self.models[label].active_model])):
            for el in ['cbtn', 'lbl', 'ebox', 'unit']:
                self.objects_main[label][self.models[label].active_model][el][idx].grid_forget()
        # Display new widgets
        self.models[label].active_model = self.objects_main[label]['active'].get()
        if self.models['IF'].active_model in ['off', 'uniform']:
            self.main['btn_fitIF_PL'].config(state='disabled')
        else:
            self.main['btn_fitIF_PL'].config(state='normal')
        for idx in range(len(self.models[label].parameters[self.models[label].active_model])):
            self.objects_main[label][self.models[label].active_model]['cbtn'][idx].grid(row=idx, column=0, sticky='e')
            self.objects_main[label][self.models[label].active_model]['lbl'][idx].grid(row=idx, column=1, sticky='e')
            self.objects_main[label][self.models[label].active_model]['ebox'][idx].grid(row=idx, column=2)
            self.objects_main[label][self.models[label].active_model]['unit'][idx].grid(row=idx, column=3, sticky='w')
        self.createTree()
        self.checkFitFixed(label)

    def checkFitFixed(self, label):
        for var, ebox in zip(self.objects_main[label][self.models[label].active_model]['var_cbtn'], self.objects_main[label][self.models[label].active_model]['ebox']):
            if var.get() == 0:
                ebox.config(state='normal')
            else:
                ebox.delete(0, 'end')
                ebox.config(state='disabled')
        all_selected = 1
        for label in list(self.models.keys()):
            for var in self.objects_main[label][self.models[label].active_model]['var_cbtn']:
                all_selected *= var.get()
        if all_selected == 0:
            self.objects_main['var_sel_all'].set(0)
        else:
            self.objects_main['var_sel_all'].set(1)

    def updateTreeData(self, tab):
        GUI = self.tabs['data'][tab]['GUI']
        values = []
        for label in list(self.models.keys()):
            mod = self.models[label].active_model
            for var in GUI.objects[label][mod]['var_sld']:
                values.append(var.get())
        values.append('{:.4f}'.format(GUI.computeFitSqDiff()))
        self.tree['tree'].insert("", 'end', text=tab, values=values)

    def fitIF_PL_All(self):
        self.createTree()
        fit_error = []
        if not self.tabs['data']:
            message = 'Please import data to fit.'
            tk.messagebox.showwarning('No data to fit', message)
            return
        for tab in self.tabs['data']:
            x_var = self.main['c_box_fit'].get()
            if x_var == 'wavelength':
                self.tabs['data'][tab]['GUI'].objects['options']['r_btn_wave'].invoke()
            elif x_var == 'energy':
                self.tabs['data'][tab]['GUI'].objects['options']['r_btn_energy'].invoke()
            for label in list(self.models.keys()):
                mod = self.models[label].active_model
                self.tabs['data'][tab]['GUI'].objects[label][mod]['r_btn'].invoke()
                for idx, (check, ebox) in enumerate(zip(self.objects_main[label][mod]['var_cbtn'], self.objects_main[label][mod]['ebox'])):
                    self.tabs['data'][tab]['GUI'].objects[label][mod]['var_cbtn'][idx].set(check.get())
                    if check.get() == 0:
                        if not ebox.get():
                            message = 'Please set value of fixed parameter.'
                            tk.messagebox.showwarning('Undefined parameter', message)
                            return
                        else:
                            self.tabs['data'][tab]['GUI'].objects[label][mod]['var_sld'][idx].set(ebox.get())
                self.tabs['data'][tab]['GUI'].updateParamFunction(label)
            try:
                self.tabs['data'][tab]['GUI'].fitIF_PL()
                self.updateTreeData(tab)
            except RuntimeError:
                fit_error.append(tab)
        if fit_error:
            message = 'One or more fits were not possible. Try adjusting the initial values or decreasing the number of fit parameters and try again.'
            message += '\nError in tab: {}'.format(fit_error[0])
            for error in fit_error[1:]:
                message += ', {}'.format(error)
            tk.messagebox.showerror('Fit error', message)

    def fitPL_All(self):
        self.createTree()
        fit_error = []
        if not self.tabs['data']:
            message = 'Please import data to fit.'
            tk.messagebox.showwarning('No data to fit', message)
            return
        for tab in self.tabs['data']:
            x_var = self.main['c_box_fit'].get()
            if x_var == 'wavelength':
                self.tabs['data'][tab]['GUI'].objects['options']['r_btn_wave'].invoke()
            elif x_var == 'energy':
                self.tabs['data'][tab]['GUI'].objects['options']['r_btn_energy'].invoke()
            for label in list(self.models.keys()):
                mod = self.models[label].active_model
                self.tabs['data'][tab]['GUI'].objects[label][mod]['r_btn'].invoke()
                for idx, (check, ebox) in enumerate(zip(self.objects_main[label][mod]['var_cbtn'], self.objects_main[label][mod]['ebox'])):
                    self.tabs['data'][tab]['GUI'].objects[label][mod]['var_cbtn'][idx].set(check.get())
                    if check.get() == 0:
                        if not ebox.get():
                            message = 'Please set value of fixed parameter.'
                            tk.messagebox.showwarning('Undefined parameter', message)
                            return
                        else:
                            self.tabs['data'][tab]['GUI'].objects[label][mod]['var_sld'][idx].set(ebox.get())
                self.tabs['data'][tab]['GUI'].updateParamFunction(label)
            try:
                self.tabs['data'][tab]['GUI'].fitPL()
                self.updateTreeData(tab)
            except RuntimeError:
                fit_error.append(tab)
        if fit_error:
            message = 'One or more fits were not possible. Try adjusting the initial values or decreasing the number of fit parameters and try again.'
            message += '\nError in tab: {}'.format(fit_error[0])
            for error in fit_error[1:]:
                message += ', {}'.format(error)
            tk.messagebox.showerror('Fit error', message)


    def createCombobox(self, window, label, initial):
        combobox = ttk.Combobox(window, values=list(self.models[label].parameters.keys()), state='readonly')
        combobox.bind('<<ComboboxSelected>>', lambda event:self.selectCombobox(event, label))
        combobox.current(list(self.models[label].parameters.keys()).index(initial))
        return combobox

    def selectCombobox(self, event, label):
        value = list(self.models[label].parameters.keys())[self.objects_main[label]['cbox'].current()]
        self.objects_main[label]['active'].set(value)
        self.updateModel(label)

    def openFile(self):
        filepath = askopenfilename(filetypes=[("CSV", "*.csv")])
        if not filepath:
            return
        filename = path.basename(filepath)
        x_var = {'μm': 'w', 'nm': 'w', 'eV': 'e'}
        try:
            self.window.title(filename)
            exp_data = pd.read_csv(filepath)
            unit = self.main['c_box_import'].get()
            if unit == 'nm':
                xdata = np.array(exp_data[exp_data.columns[0]])*1e-9
            elif unit == 'μm':
                xdata = np.array(exp_data[exp_data.columns[0]])*1e-6
            elif unit == 'eV':
                xdata = np.array(exp_data[exp_data.columns[0]])
            for col in exp_data.columns[1:]:
                new_tab = ttk.Frame(self.tab_control)
                ydata = np.array(exp_data[col])
                ydata /= np.max(ydata)
                GUI = self.initialiseTab(new_tab, col, xdata, ydata,
                                         x_var[unit])
                self.tab_control.add(new_tab, text=col)
                self.tab_control.pack(expand=1, fill='both')
                self.tabs['data'][col] = {'frame': new_tab, 'GUI': GUI}
            sum_tab = ttk.Frame(self.tab_control)
            self.tabs['summary'] = {'frame': sum_tab, 'GUI': SummaryTab(sum_tab, self.tabs['data'], x_var[unit], xdata, self.c_data, self.opt_data)}
            self.main['btn_open'].config(state='disabled')
            self.main['btn_config'].config(state='disabled')
        except:
            self.window.title('PL Modeling Program')
            message = 'Error importing data. Check the file format and try again.'
            tk.messagebox.showerror('Import error', message)

    def createIFPLModels(self, xdata_w, xdata_e):
        # Define system parameters
        IF_d = self.c_data['IF']['thickness']
        IF_wave_laser = self.c_data['IF']['wave_laser']
        IF_k_Mo = self.c_data['IF']['k']
        IF_roughness = self.c_data['IF']['R_rms']

        # Interference Function parameters
        IF_delta_pos = Parameter(value=self.c_data['IF']['delta'][1],
                                 bounds=(self.c_data['IF']['delta'][0],
                                         self.c_data['IF']['delta'][2]),
                                 s_label='δ(λ-x)', s_unit='nm',
                                 s_scale=1e-9, s_prec=0)
        IF_gauss_mean = Parameter(value=self.c_data['IF']['mu'][1],
                                  bounds=(self.c_data['IF']['mu'][0],
                                          self.c_data['IF']['mu'][2]),
                                  s_label='μ', s_unit='nm',
                                  s_scale=1e-9, s_prec=0)
        IF_gauss_std = Parameter(value=self.c_data['IF']['sigma'][1],
                                 bounds=(self.c_data['IF']['sigma'][0],
                                         self.c_data['IF']['sigma'][2]),
                                 s_label='σ', s_unit='nm',
                                 s_scale=1e-9, s_prec=0)

        # Bandgap fluctuations model parameters
        BGF_mean_E_g = Parameter(value=self.c_data['BGF']['E_g'][1],
                                 bounds=(self.c_data['BGF']['E_g'][0],
                                         self.c_data['BGF']['E_g'][2]),
                                 s_label='E_g', s_unit='eV',
                                 s_scale=1, s_prec=3)
        BGF_beta = Parameter(value=self.c_data['BGF']['beta'][1],
                             bounds=(self.c_data['BGF']['beta'][0],
                                     self.c_data['BGF']['beta'][2]),
                             s_label='β', s_unit='-',
                             s_scale=1, s_prec=2)
        BGF_sigma_g = Parameter(value=self.c_data['BGF']['sigma_g'][1],
                                bounds=(self.c_data['BGF']['sigma_g'][0],
                                        self.c_data['BGF']['sigma_g'][2]),
                                s_label='σ_g', s_unit='meV',
                                s_scale=1e-3, s_prec=0)
        BGF_T = Parameter(value=self.c_data['BGF']['T'][1],
                          bounds=(self.c_data['BGF']['T'][0],
                                  self.c_data['BGF']['T'][2]),
                          s_label='T', s_unit='K',
                          s_scale=1, s_prec=0)

        # Electrostatic fluctuations model parameters
        EF_E_g = Parameter(value=self.c_data['EF']['E_g'][1],
                           bounds=(self.c_data['EF']['E_g'][0],
                                   self.c_data['EF']['E_g'][2]),
                           s_label='E_g', s_unit='eV',
                           s_scale=1, s_prec=3)
        EF_theta = Parameter(value=self.c_data['EF']['theta'][1],
                             bounds=(self.c_data['EF']['theta'][0],
                                     self.c_data['EF']['theta'][2]),
                             s_label='θ', s_unit='-',
                             s_scale=1, s_prec=2)
        EF_gamma = Parameter(value=self.c_data['EF']['gamma'][1],
                             bounds=(self.c_data['EF']['gamma'][0],
                                     self.c_data['EF']['gamma'][2]),
                             s_label='γ', s_unit='meV',
                             s_scale=1e-3, s_prec=0)
        EF_dmu = Parameter(value=self.c_data['EF']['delta_mu'][1],
                           bounds=(self.c_data['EF']['delta_mu'][0],
                                   self.c_data['EF']['delta_mu'][2]),
                           s_label='Δμ', s_unit='eV',
                           s_scale=1, s_prec=2)
        EF_T = Parameter(value=self.c_data['EF']['T'][1],
                         bounds=(self.c_data['EF']['T'][0],
                                 self.c_data['EF']['T'][2]),
                         s_label='T', s_unit='K',
                         s_scale=1, s_prec=0)
        EF_a0d = Parameter(value=self.c_data['EF']['a_0d'][1],
                           bounds=(self.c_data['EF']['a_0d'][0],
                                   self.c_data['EF']['a_0d'][2]),
                           s_label='a_0d', s_unit='-',
                           s_scale=1, s_prec=1)

        # Unified potential fluctuations model parameters
        UPF_mean_E_g = Parameter(value=self.c_data['UPF']['E_g'][1],
                                 bounds=(self.c_data['UPF']['E_g'][0],
                                         self.c_data['UPF']['E_g'][2]),
                                 s_label='E_g', s_unit='eV',
                                 s_scale=1, s_prec=3)
        UPF_beta = Parameter(value=self.c_data['UPF']['beta'][1],
                             bounds=(self.c_data['UPF']['beta'][0],
                                     self.c_data['UPF']['beta'][2]),
                             s_label='β', s_unit='-',
                             s_scale=1, s_prec=2)
        UPF_sigma_g = Parameter(value=self.c_data['UPF']['sigma_g'][1],
                                bounds=(self.c_data['UPF']['sigma_g'][0],
                                        self.c_data['UPF']['sigma_g'][2]),
                                s_label='σ_g', s_unit='meV',
                                s_scale=1e-3, s_prec=0)
        UPF_theta = Parameter(value=self.c_data['UPF']['theta'][1],
                              bounds=(self.c_data['UPF']['theta'][0],
                                      self.c_data['UPF']['theta'][2]),
                              s_label='θ', s_unit='-',
                              s_scale=1, s_prec=2)
        UPF_gamma = Parameter(value=self.c_data['UPF']['gamma'][1],
                              bounds=(self.c_data['UPF']['gamma'][0],
                                      self.c_data['UPF']['gamma'][2]),
                              s_label='γ', s_unit='meV',
                              s_scale=1e-3, s_prec=0)
        UPF_T = Parameter(value=self.c_data['UPF']['T'][1],
                          bounds=(self.c_data['UPF']['T'][0],
                                  self.c_data['UPF']['T'][2]),
                          s_label='T', s_unit='K',
                          s_scale=1, s_prec=0)
        UPF_a0d = Parameter(value=self.c_data['UPF']['a_0d'][1],
                            bounds=(self.c_data['UPF']['a_0d'][0],
                                    self.c_data['UPF']['a_0d'][2]),
                            s_label='a_0d', s_unit='-',
                            s_scale=1, s_prec=1)

        # Define the classes needed for the model
        class_IF = IF(d=IF_d, wave_laser=IF_wave_laser, wavelengths=xdata_w,
                      luminescence='delta', opt_data=self.opt_data,
                      params=[IF_delta_pos.value], rough=IF_roughness,
                      k_Mo=IF_k_Mo)
        class_PL = PLModel(params=[BGF_mean_E_g.value, BGF_beta.value,
                                   BGF_sigma_g.value, BGF_T.value],
                           model='BGF', E=xdata_e)

        # Define the models based on the defined classes
        model_IF = Model(class_IF, parameters={'uniform': [],
                                               'delta': [IF_delta_pos],
                                               'gaussian': [IF_gauss_mean,
                                                            IF_gauss_std],
                                               'off': []},
                         x_var='w', label='Interference Function',
                         active_model='delta')
        model_PL = Model(class_PL, parameters={'BGF': [BGF_mean_E_g, BGF_beta,
                                                       BGF_sigma_g, BGF_T],
                                               'EF': [EF_E_g, EF_theta,
                                                      EF_gamma, EF_dmu, EF_T,
                                                      EF_a0d],
                                               'UPF': [UPF_mean_E_g, UPF_beta,
                                                       UPF_sigma_g, UPF_theta,
                                                       UPF_gamma, UPF_T,
                                                       UPF_a0d]},
                         x_var='w', label='PL Modeling',
                         active_model='BGF')
        return {'IF': model_IF, 'PL': model_PL}

    def initialiseTab(self, window, name, xdata, ydata, x_var):
        if x_var == 'w':
            xdata_e, ydata_e = W2E(xdata, ydata)
            ydata_e /= np.max(ydata_e)
            models = self.createIFPLModels(xdata, xdata_e)
            GUI = InteractiveFittingTK(window, name=name, models=models,
                                       data={'e': [xdata_e, ydata_e],
                                             'w': [xdata, ydata]})
        elif x_var == 'e':
            xdata_w, ydata_w = E2W(xdata, ydata)
            ydata_w /= np.max(ydata_w)
            models = self.createIFPLModels(xdata_w, xdata)
            GUI = InteractiveFittingTK(window, name=name, models=models,
                                       data={'e': [xdata, ydata],
                                             'w': [xdata_w, ydata_w]})
        return GUI


class InteractiveFittingTK:
    """Class that defines the interactive fitting pane for a data tab."""
    model_w = 100
    label_w = 50
    scale_w = 200
    spin_box_w = 5
    unit_w = 40
    fit_w = 30

    # def __init__(self, figure, axes, graphs, models, data, x_plot, x_scale):
    def __init__(self, window, name, models, data):
        # Tkinter
        self.window = window
        self.name = name
        self.models = models
        self.data = data
        self.x_scale = {'e': 1, 'w': 1e6}
        self.font_title = tkFont.Font(weight='bold', size=18)
        self.font_legend = tkFont.Font(weight='bold', size=11)
        self.font_label = tkFont.Font(size=11)

        self.current_axis = 1

        # Format sizes
        self.window.rowconfigure(0, minsize=200, weight=1)
        self.window.rowconfigure(2, minsize=250, weight=1)
        self.window.rowconfigure(4, minsize=100, weight=1)
        self.window.columnconfigure(0, minsize=450, weight=1)

        # Define frames
        self.frm_bar = tk.Frame(master=self.window)
        self.frm_IF = tk.Frame(master=self.window)
        self.frm_PL = tk.Frame(master=self.window)
        self.frm_fit = tk.Frame(master=self.window)

        # Plots
        self.figure, self.axes, self.graphs, self.toolbar = self.createPlot()

        # Interference and PL objects
        self.objects = {}
        self.objects['IF'] = self.createFrameObjects(frame=self.frm_IF, title='Interference Correction', label='IF')
        self.objects['PL'] = self.createFrameObjects(frame=self.frm_PL, title='PL Modeling', label='PL')

        # Fitting objects
        self.objects['fit'] = self.createFitObjects(frame=self.frm_fit, title='Fit')

        # Extra options
        self.objects['options'] = self.createOptionObjects(frame=self.frm_fit, title='Options')

        # Keep copy of params to avoid rounding errors when changing models
        self.copy_params = {}
        for label in self.models:
            params_dict = {}
            for model in self.models[label].parameters:
                model_params = [param.value for param in self.models[label].parameters[model]]
                params_dict[model] = model_params
            self.copy_params[label] = params_dict

        # Set right models
        self.objects['IF']['active'].set('off')
        self.updateModel('IF')
        self.updateModel('PL')

        # Set all frames
        self.frm_bar.grid(row=5, column=0, sticky='news')
        self.frm_IF.grid(row=0, column=1, sticky='news')
        ttk.Separator(master=self.window, orient=tk.HORIZONTAL).grid(row=1, column=1, pady=10, sticky='news')
        self.frm_PL.grid(row=2, column=1, sticky='news')
        ttk.Separator(master=self.window, orient=tk.HORIZONTAL).grid(row=3, column=1, pady=10, sticky='news')
        self.frm_fit.grid(row=4, column=1, sticky='nw')

    def createPlot(self):
        figure = Figure(figsize=(5, 5))
        ax_top = figure.add_subplot(211)
        ax_top.xaxis.set_ticklabels([])
        ax_bot = figure.add_subplot(212)
        canvas = FigureCanvasTkAgg(figure, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, rowspan=5, sticky='news')

        toolbar = NavigationToolbar2Tk(canvas=canvas, window=self.frm_bar)
        toolbar.update()
        canvas._tkcanvas.grid()

        data_graph, = ax_top.plot(self.data['w'][0]*self.x_scale['w'], self.data['w'][1], color='k',
                                  label='data', linewidth=1.2)
        IF_graph, = ax_top.plot(self.data['w'][0]*self.x_scale['w'], self.models['IF'].model_class.IF, label='IF', linewidth=1.2)

        corrected_PL = self.data['w'][1]/self.models['IF'].model_class.IF
        corrected_PL /= np.max(corrected_PL)
        BGF_w = E2W(self.data['e'][0], self.models['PL'].model_class.emission)[1]
        BGF_w /= np.max(BGF_w)

        data_fix_graph, = ax_bot.plot(self.data['w'][0]*self.x_scale['w'], corrected_PL,
                                      label='corrected data', color='k', linewidth=1.2)
        PL_graph, = ax_bot.plot(self.data['w'][0]*self.x_scale['w'], BGF_w,
                                label='PL model', linestyle='-', linewidth=1.2)

        ax_top.set_position([0.13, 0.55, 0.8, 0.42])
        ax_bot.set_position([0.13, 0.10, 0.8, 0.42])
        ax_bot.set_xlabel('Wavelength [μm]')
        for ax in [ax_top, ax_bot]:
            ax.legend(prop={'size': 9})
            ax.set_ylabel('Normalised PL [-]')
            ax.set_ylim(-.05, 1.05)
            ax.set_xlim(np.min(self.data['w'][0]*self.x_scale['w']), np.max(self.data['w'][0]*self.x_scale['w']))
        return figure, [ax_top, ax_bot], [data_graph, IF_graph, data_fix_graph, PL_graph], toolbar

    def saveModelData(self):
        filepath = asksaveasfilename(defaultextension='csv', filetypes=[("CSV", "*.csv"), ("All Files", "*.*")])
        if not filepath:
            return
        output_dict = {}
        output_dict['Data'] = self.name
        for label in list(self.models.keys()):
            params = self.objects[label][self.models[label].active_model]
            for (lbl, unit, val) in zip(params['lbl'], params['unit'], params['var_sld']):
                output_dict['{} [{}]'.format(lbl['text'], unit['text'])] = [val.get()]
        output_dict['Model error'] = '{:.4f}'.format(self.computeFitSqDiff())
        output = pd.DataFrame(output_dict)
        try:
            output.to_csv(filepath, index=False, encoding="utf-8-sig")
            message = 'Model data saved to {}.'.format(filepath)
            tk.messagebox.showinfo('Model data saved successfully', message)
        except:
            message = 'Not able to save data to {}. The file might be open.'.format(filepath)
            tk.messagebox.showerror('Save error', message)

    def exportPlotData(self):
        filepath = asksaveasfilename(defaultextension='csv', filetypes=[("CSV", "*.csv"), ("All Files", "*.*")])
        if not filepath:
            return
        graph_data = {self.axes[1].get_xlabel(): self.graphs[0].get_xdata()}
        graph_data = {**graph_data, **self.getPlotData()}
        graph_data = pd.DataFrame(graph_data)
        try:
            graph_data.to_csv(filepath, index=False, encoding="utf-8-sig")
            message = 'Plot data saved to {}.'.format(filepath)
            tk.messagebox.showinfo('Plot data saved successfully', message)
        except:
            message = 'Not able to save data to {}. The file might be open.'.format(filepath)
            tk.messagebox.showerror('Save error', message)

    def getPlotData(self):
        graph_data = {}
        for graph in self.graphs:
            graph_data['{}_{}'.format(self.name, graph.get_label())] = graph.get_ydata()
        return graph_data

    def createOptionObjects(self, frame, title):
        frame.columnconfigure(2, minsize=150)
        # Title
        (tk.Label(master=frame, text=title, font=self.font_title)
         .grid(row=0, column=2, sticky='w'))
        btn_save_model = tk.Button(master=frame, text='Save model values', font=self.font_label, command=self.saveModelData)
        btn_save_model.grid(row=1, column=2, sticky='w', pady=5)
        btn_export_plot = tk.Button(master=frame, text='Export plot data', font=self.font_label, command=self.exportPlotData)
        btn_export_plot.grid(row=2, column=2, sticky='w', pady=5)
        (tk.Label(master=frame, text='x-axis variable', font=self.font_legend)
         .grid(row=1, column=3, sticky='w', pady=5, padx=5))
        strvar_x = tk.StringVar()
        strvar_x.set('w')
        r_btn_wave = tk.Radiobutton(master=frame, text='wavelength', variable=strvar_x, value='w', font=self.font_label, command=self.updateAxisUnit)
        r_btn_wave.grid(row=2, column=3, sticky='w')
        r_btn_energy = tk.Radiobutton(master=frame, text='energy', variable=strvar_x, value='e', font=self.font_label, command=self.updateAxisUnit)
        r_btn_energy.grid(row=3, column=3, sticky='w')
        objects = {'var_x': strvar_x, 'r_btn_wave': r_btn_wave, 'r_btn_energy': r_btn_energy}
        return objects

    def createFitObjects(self, frame, title):
        frame.columnconfigure(0, minsize=60)
        frame.columnconfigure(1, minsize=100)
        # Title
        (tk.Label(master=frame, text=title, font=self.font_title)
         .grid(row=0, column=0, sticky='w'))
        btn_fitPL = tk.Button(master=frame, text='Fit PL', font=self.font_label, command=lambda:self.fitFuncCatch(self.fitPL))
        btn_fitPL.grid(row=1, column=0, sticky='w', pady=5)
        btn_fitIFPL = tk.Button(master=frame, text='Fit IF & PL', font=self.font_label, command=lambda:self.fitFuncCatch(self.fitIF_PL))
        btn_fitIFPL.grid(row=1, column=1, sticky='w', pady=5)
        text = 'Model error: {:.4f}'.format(self.computeFitSqDiff())
        error_txt = tk.StringVar()
        error_txt.set(text)
        error_lbl = tk.Label(master=frame, textvariable=error_txt, font=self.font_label)
        error_lbl.grid(row=2, column=0, columnspan=2, sticky='w', pady=5)
        var_select_all = tk.BooleanVar()
        cbtn_select_all = ttk.Checkbutton(master=frame, variable=var_select_all, command=self.selectAllFit, text='Select all')
        cbtn_select_all.grid(row=3, column=0, columnspan=2, sticky='w', pady=5)
        objects = {'btnPL': btn_fitPL, 'btnIF_PL': btn_fitIFPL, 'var_sel_all': var_select_all, 'err_txt': error_txt}
        return objects

    def selectAllFit(self):
        if self.objects['fit']['var_sel_all'].get() == 1:
            set_val = 1
        else:
            set_val = 0
        for label in list(self.models.keys()):
            for var in self.objects[label][self.models[label].active_model]['var_cbtn']:
                var.set(set_val)


    def updateModel(self, label):
        # Copy old model values
        self.copy_params[label][self.models[label].active_model] = self.models[label].model_class.params
        # Update display
        # Remove present widgets
        self.objects[label][self.models[label].active_model]['leg'][0].grid_forget()
        self.objects[label][self.models[label].active_model]['leg'][1].grid_forget()
        for idx in range(len(self.models[label].parameters[self.models[label].active_model])):
            for el in ['lbl', 'sld', 'sbox', 'cbtn', 'unit']:
                self.objects[label][self.models[label].active_model][el][idx].grid_forget()
        # Display new widgets
        self.models[label].active_model = self.objects[label]['active'].get()
        if label == 'PL':
            self.objects[label][self.models[label].active_model]['leg'][0].grid(row=2, column=2, columnspan=3, sticky='w')
            self.objects[label][self.models[label].active_model]['leg'][1].grid(row=2, column=5, sticky='news')
        elif label == 'IF':
            if self.models[label].active_model not in ['off', 'uniform']:
                self.objects[label][self.models[label].active_model]['leg'][0].grid(row=2, column=2, columnspan=3, sticky='w')
                self.objects[label][self.models[label].active_model]['leg'][1].grid(row=2, column=5, sticky='news')
                self.objects['fit']['btnIF_PL'].config(state='normal')
            else:
                self.objects['fit']['btnIF_PL'].config(state='disabled')
        for idx in range(len(self.models[label].parameters[self.models[label].active_model])):
            self.objects[label][self.models[label].active_model]['lbl'][idx].grid(row=idx+3, column=1, sticky='e')
            self.objects[label][self.models[label].active_model]['sld'][idx].grid(row=idx+3, column=2)
            self.objects[label][self.models[label].active_model]['sbox'][idx].grid(row=idx+3, column=3)
            self.objects[label][self.models[label].active_model]['unit'][idx].grid(row=idx+3, column=4, sticky='w')
            self.objects[label][self.models[label].active_model]['cbtn'][idx].grid(row=idx+3, column=5, sticky='e')
        # Update model parameters
        self.models[label].model_class.params = self.copy_params[label][self.models[label].active_model]
        # Update class model
        self.models[label].model_class.updateModel(self.models[label].active_model)
        # Update select all button
        self.checkFit()
        # Update plot
        self.updateGraph()

    def createFrameObjects(self, frame, title, label):
        dict_mod = {}
        # Title
        (tk.Label(master=frame, text=title, font=self.font_title)
         .grid(row=0, column=0, columnspan=5, sticky='w'))
        # Legend
        (tk.Label(master=frame, text='Model', font=self.font_legend)
         .grid(row=2, column=0, sticky='w'))
        # Min grid size
        for col, size in zip([0, 1, 2, 3, 4, 5],[self.model_w, self.label_w, self.scale_w, self.spin_box_w, self.unit_w, self.fit_w]):
            frame.grid_columnconfigure(col, minsize=size)
        # Define parameter widgets
        model = self.models[label]
        strvar_mod = tk.StringVar()
        strvar_mod.set(model.active_model)
        dict_mod['active'] = strvar_mod
        for idx, mod in enumerate(model.parameters.keys()):
            # Initialise containers
            objects = {}
            lbl_param = []
            sld_param = []
            var_sld_param = []
            sbox_param = []
            unit_param = []
            cbtn_param = []
            var_cbtn_param = []
            # Column legends
            lbl_parameter = tk.Label(master=frame, text='Parameter', font=self.font_legend)
            lbl_fit = tk.Label(master=frame, text='Fit', font=self.font_legend)
            objects['leg'] = [lbl_parameter, lbl_fit]
            r_btn = tk.Radiobutton(master=frame, text=mod, variable=strvar_mod, value=mod, font=self.font_label, command=lambda:self.updateModel(label))
            objects['r_btn'] = r_btn
            r_btn.grid(row=idx+3, column=0, sticky='w')
            # Labels, sliders, spinboxes and checkbuttons
            for idx, param in enumerate(model.parameters[mod]):
                # Label
                lbl = tk.Label(master=frame, text=param.s_label, font=self.font_label)
                lbl_param.append(lbl)
                # Slider
                var_param = tk.IntVar(value=param.value/param.s_scale) if param.s_prec == 0 else tk.DoubleVar(value=param.value/param.s_scale)
                sld = Slider(master=frame, variable=var_param, orient='horizontal', command=lambda *args:self.updateParamFunction(label, *args), length=self.scale_w, from_=param.bounds[0]/param.s_scale, to=param.bounds[1]/param.s_scale, precision=param.s_prec)
                sld_param.append(sld)
                var_sld_param.append(var_param)
                # Spinbox
                sbox = tk.Spinbox(master=frame, textvariable=var_param, command=lambda *args:self.updateParamFunction(label, *args), width=self.spin_box_w, from_=param.bounds[0]/param.s_scale, to=param.bounds[1]/param.s_scale, increment=0.1**(param.s_prec))
                sbox.bind('<Return>', lambda *args:self.updateParamFunction(label, *args))
                sbox_param.append(sbox)
                # Unit
                unit = tk.Label(master=frame, text=param.s_unit, font=self.font_label)
                unit_param.append(unit)
                # Checkbutton
                intvar_check = tk.BooleanVar()
                intvar_check.set(True)
                cbtn = ttk.Checkbutton(master=frame, variable=intvar_check, command=self.checkFit)
                cbtn_param.append(cbtn)
                var_cbtn_param.append(intvar_check)
            objects['lbl'] = lbl_param
            objects['sld'] = sld_param
            objects['var_sld'] = var_sld_param
            objects['sbox'] = sbox_param
            objects['unit'] = unit_param
            objects['cbtn'] = cbtn_param
            objects['var_cbtn'] = var_cbtn_param
            dict_mod[mod] = objects
        return dict_mod

    def checkFit(self):
        all_selected = 1
        for label in list(self.models.keys()):
            for var in self.objects[label][self.models[label].active_model]['var_cbtn']:
                all_selected *= var.get()
        if all_selected == 0:
            self.objects['fit']['var_sel_all'].set(0)
        else:
            self.objects['fit']['var_sel_all'].set(1)


    def updateParamFunction(self, label, *args):
        for idx, (slider, parameter) in enumerate(zip(self.objects[label][self.models[label].active_model]['sld'], self.models[label].parameters[self.models[label].active_model])):
            parameter.value = slider.get()*parameter.s_scale
            self.models[label].model_class.params[idx] = parameter.value
        self.updateGraph()

    def computeFitSqDiff(self):
        return np.sum((self.graphs[2].get_ydata()
                      - self.graphs[3].get_ydata())**2
                      / self.graphs[3].get_ydata().size)**0.5

    def fitFuncCatch(self, func):
        try:
            func()
        except RuntimeError:
            message = 'Fit not possible. Try adjusting the initial values or decreasing the number of fit parameters and fit again.'
            tk.messagebox.showerror('Fit error', message)

    def fitInit(self, label):
        vars_bool = [var.get() for var in self.objects[label][self.models[label].active_model]['var_cbtn']]
        fixed_bool = [not val for val in vars_bool]
        vars = list(compress(self.models[label].parameters[self.models[label].active_model], vars_bool))
        vars_lb = [param.bounds[0] for param in vars]
        vars_ub = [param.bounds[1] for param in vars]
        vars_bounds = [vars_lb, vars_ub]
        vars_idx = np.cumsum(np.array(vars_bool).astype(int))-1
        params_values = list(self.models[label].model_class.params)
        vars_values = list(np.array(params_values)[vars_bool])
        vars_values = [vars_lb[idx] if vars_values[idx] < vars_lb[idx] else vars_values[idx] for idx in range(len(vars_values))]
        vars_values = [vars_ub[idx] if vars_values[idx] > vars_ub[idx] else vars_values[idx] for idx in range(len(vars_values))]
        return params_values, vars_values, vars_bounds, fixed_bool, vars_bool, vars_idx

    def fitPL(self):
        ydata = W2E(self.models['IF'].model_class.wavelengths, self.data['w'][1]/self.models['IF'].model_class.IF)[1]
        ydata /= np.max(ydata)
        params, vars_values, vars_bounds, fixed_bool, vars_bool, vars_idx = self.fitInit('PL')
        fit = curve_fit(lambda energy, *args: self.models['PL'].model_class.fitFunction(energy, *[params[idx] if fixed else args[not_fixed] for (idx, (fixed, not_fixed)) in enumerate(zip(fixed_bool, vars_idx))]), self.models['PL'].model_class.E, ydata,
                        p0=vars_values, bounds=vars_bounds)
        new_params = np.array(params)
        new_params[vars_bool] = fit[0]
        for idx, (slider, param) in enumerate(zip(self.objects['PL'][self.models['PL'].active_model]['sld'], self.models['PL'].parameters[self.models['PL'].active_model])):
            slider.set(new_params[idx]/param.s_scale)
        self.models['PL'].model_class.params = new_params
        self.updateGraph()

    def fitIF_PL(self):
        params_IF, vars_values_IF, vars_bounds_IF, fixed_bool_IF, vars_bool_IF, vars_idx_IF = self.fitInit('IF')
        params_PL, vars_values_PL, vars_bounds_PL, fixed_bool_PL, vars_bool_PL, vars_idx_PL = self.fitInit('PL')
        params = params_IF+params_PL
        vars_values = vars_values_IF+vars_values_PL
        vars_bounds = [vars_bounds_IF[0]+vars_bounds_PL[0], vars_bounds_IF[1]+vars_bounds_PL[1]]
        fixed_bool = fixed_bool_IF+fixed_bool_PL
        vars_bool = vars_bool_IF+vars_bool_PL
        vars_idx = np.concatenate((vars_idx_IF, vars_idx_PL+1+vars_idx_IF[-1]))
        num_params_IF = len(params_IF)
        fit = curve_fit(lambda energy, *args: self.fitFunctionIF_PL(energy, num_params_IF, *[params[idx] if fixed else args[not_fixed] for (idx, (fixed, not_fixed)) in enumerate(zip(fixed_bool, vars_idx))]), self.models['PL'].model_class.E, self.data['w'][1],
                        p0=vars_values, bounds=vars_bounds)
        new_params = np.array(params)
        new_params[vars_bool] = fit[0]
        for idx, (slider, param) in enumerate(zip(self.objects['IF'][self.models['IF'].active_model]['sld'], self.models['IF'].parameters[self.models['IF'].active_model])):
            slider.set(new_params[idx]/param.s_scale)
        for idx, (slider, param) in enumerate(zip(self.objects['PL'][self.models['PL'].active_model]['sld'], self.models['PL'].parameters[self.models['PL'].active_model])):
            slider.set(new_params[idx+num_params_IF]/param.s_scale)
        self.models['IF'].model_class.params = new_params[:num_params_IF]
        self.models['PL'].model_class.params = new_params[num_params_IF:]
        self.updateGraph()

    def fitFunctionIF_PL(self, E, num_params_IF, *params):
        self.models['IF'].model_class.params = list(params[:num_params_IF])
        self.models['IF'].model_class.solve()
        PL_model = self.models['PL'].model_class.fitFunction(E, *params[num_params_IF:])
        PL_w = E2W(E, PL_model)[1]
        y_model = PL_w*self.models['IF'].model_class.IF
        return y_model/np.max(y_model)

    def updateAxisUnit(self):
        label = self.objects['options']['var_x'].get()
        if ((label == 'w' and self.current_axis == -1) or
           (label == 'e' and self.current_axis == 1)):
            self.current_axis *= -1
            self.updateGraph()
            for graph in self.graphs:
                graph.set_xdata(self.data[label][0]*self.x_scale[label])
            self.axes[0].set_xlim(np.min(self.data[label][0]*self.x_scale[label]), np.max(self.data[label][0]*self.x_scale[label]))
            self.axes[1].set_xlim(np.min(self.data[label][0]*self.x_scale[label]), np.max(self.data[label][0]*self.x_scale[label]))
            if label == 'w':
                self.graphs[0].set_ydata(self.data['w'][1])
                self.axes[1].set_xlabel('Wavelength [μm]')
            elif label == 'e':
                self.graphs[0].set_ydata(self.data['e'][1])
                self.axes[1].set_xlabel('Energy [eV]')

    def updateGraph(self):
        if self.models['IF'].active_model == 'off':
            self.models['IF'].model_class.IF = np.ones(self.models['IF'].model_class.IF.shape)
        else:
            self.models['IF'].model_class.solve()
        self.models['PL'].model_class.solve()
        IF = self.models['IF'].model_class.IF
        corrected_PL = self.data['w'][1]/self.models['IF'].model_class.IF
        PL_model = self.models['PL'].model_class.emission
        if self.current_axis == 1:
            self.graphs[1].set_ydata(IF)
            self.graphs[2].set_ydata(corrected_PL/np.max(corrected_PL))
            PL_w = E2W(self.models['PL'].model_class.E, PL_model)[1]
            self.graphs[3].set_ydata(PL_w/np.max(PL_w))
        elif self.current_axis == -1:
            IF_e = W2E(self.models['IF'].model_class.wavelengths, IF)[1]
            self.graphs[1].set_ydata(IF_e/np.max(IF_e))
            corrected_PL_e = W2E(self.models['IF'].model_class.wavelengths,
                                 corrected_PL)[1]
            self.graphs[2].set_ydata(corrected_PL_e/np.max(corrected_PL_e))
            self.graphs[3].set_ydata(PL_model/np.max(PL_model))
        self.objects['fit']['err_txt'].set('Model error: {:.4f}'.format(self.computeFitSqDiff()))
        self.figure.canvas.draw_idle()


class SummaryTab(InteractiveFittingTK):
    """Class that define the IF analysis pane."""
    ebox_width = 5
    def __init__(self, window, data_tabs, x_var, xdata, c_data, opt_data):
        # Tkinter
        self.window = window
        self.data_tabs = data_tabs
        if x_var == 'w':
            self.x_data = {'e': W2E(xdata, np.ones(xdata.shape))[0] ,'w': xdata}
        elif x_var == 'e':
            self.x_data = {'e': xdata, 'w': E2W(xdata, np.ones(xdata.shape))[0]}
        self.x_scale = {'e': 1, 'w': 1e6}
        self.c_data = c_data
        self.opt_data = opt_data
        models = NotebookTabs.createIFPLModels(self, self.x_data['w'], self.x_data['e'])
        self.models = {'IF': models['IF']}
        self.models_PL = models['PL']
        self.font_title = tkFont.Font(weight='bold', size=18)
        self.font_legend = tkFont.Font(weight='bold', size=11)
        self.font_label = tkFont.Font(size=11)
        self.colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728',
                       u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f',
                       u'#bcbd22', u'#17becf']

        self.current_axis = 1

        # Format sizes
        self.window.rowconfigure(0, minsize=200, weight=1)
        self.window.rowconfigure(2, minsize=250, weight=1)
        self.window.rowconfigure(4, minsize=100, weight=1)
        self.window.grid_columnconfigure(0, minsize=450, weight=1)

        # Define frames
        self.frm_bar = tk.Frame(master=self.window)
        self.frm_IF = tk.Frame(master=self.window)
        self.frm_PL = tk.Frame(master=self.window)
        self.frm_fit = tk.Frame(master=self.window)

        # Plots
        self.figure, self.axes, self.graphs, self.toolbar = self.createPlot()

        # Interference and PL objects
        self.objects = {}
        self.objects['IF'] = self.createFrameObjects(frame=self.frm_IF, title='Interference Correction', label='IF')
        self.objects_PL = self.createFrameObjectsPL(frame=self.frm_PL, title='PL Modeling')

        # Fitting objects
        self.objects['fit'] = self.createFitObjects(frame=self.frm_fit, title='Fit')

        # Extra options
        self.objects['options'] = self.createOptionObjects(frame=self.frm_fit, title='Options')
        self.frm_fit.columnconfigure(3, minsize=174)

        # Keep copy of params to avoid rounding errors when changing models
        self.copy_params = {}
        for label in self.models:
            params_dict = {}
            for model in self.models[label].parameters:
                model_params = [param.value for param in self.models[label].parameters[model]]
                params_dict[model] = model_params
            self.copy_params[label] = params_dict

        # Set right models
        self.updateModel('IF')
        self.updateModelPL()

        # Set all frames
        self.frm_bar.grid(row=5, column=0, sticky='news')
        self.frm_IF.grid(row=0, column=1, sticky='news')
        ttk.Separator(master=self.window, orient=tk.HORIZONTAL).grid(row=1, column=1, pady=10, sticky='news')
        self.frm_PL.grid(row=2, column=1, sticky='news')
        ttk.Separator(master=self.window, orient=tk.HORIZONTAL).grid(row=3, column=1, pady=10, sticky='news')
        self.frm_fit.grid(row=4, column=1, sticky='nw')


        self.figure.canvas.draw_idle()

    def createFitFixedPL(self, frame, label):
        # Define parameter widgets
        model = self.models[label]
        strvar_mod = tk.StringVar()
        strvar_mod.set(list(model.parameters.keys())[self.objects_main[label]['cbox'].current()])
        self.objects_main[label]['active'] = strvar_mod
        for idx, mod in enumerate(model.parameters.keys()):
            # Initialise containers
            objects = {}
            cbtn_param = []
            var_cbtn_param = []
            lbl_param = []
            ebox_param = []
            var_ebox_param = []
            unit_param = []
            # Checkbuttons, labels, entryboxes and units
            for idx, param in enumerate(model.parameters[mod]):
                # Checkbutton
                intvar_check = tk.BooleanVar()
                intvar_check.set(True)
                cbtn = ttk.Checkbutton(master=frame, variable=intvar_check)
                cbtn_param.append(cbtn)
                var_cbtn_param.append(intvar_check)
                # Label
                lbl = tk.Label(master=frame, text=param.s_label+': ', font=self.font_label)
                lbl_param.append(lbl)
                # Entrybox
                strvar_ebox = tk.StringVar()
                ebox = tk.Entry(master=frame, textvariable=strvar_ebox, width=self.ebox_width)
                ebox_param.append(ebox)
                var_ebox_param.append(strvar_ebox)
                # Unit
                unit = tk.Label(master=frame, text=param.s_unit, font=self.font_label)
                unit_param.append(unit)
            objects['cbtn'] = cbtn_param
            objects['var_cbtn'] = var_cbtn_param
            objects['lbl'] = lbl_param
            objects['ebox'] = ebox_param
            objects['var_ebox'] = var_ebox_param
            objects['unit'] = unit_param
            self.objects_main[label][mod] = objects

    def createFrameObjectsPL(self, frame, title):
        dict_mod = {}
        # Title
        (tk.Label(master=frame, text=title, font=self.font_title)
         .grid(row=0, column=0, columnspan=5, sticky='w'))
        # Legend
        (tk.Label(master=frame, text='Model', font=self.font_legend)
         .grid(row=1, column=0, sticky='w'))
        # Column legends
        (tk.Label(master=frame, text='Parameter', font=self.font_legend)
         .grid(row=1, column=1, columnspan=3))
        (tk.Label(master=frame, text='Fit', font=self.font_legend)
         .grid(row=1, column=4, sticky='w'))
        # Min grid size
        for col, size in zip([0, 1, 3, 4],[self.model_w, self.label_w, self.unit_w, self.fit_w]):
            frame.grid_columnconfigure(col, minsize=size)
        # Define parameter widgets
        model = self.models_PL
        strvar_mod = tk.StringVar()
        strvar_mod.set(model.active_model)
        dict_mod['active'] = strvar_mod
        for idx, mod in enumerate(model.parameters.keys()):
            # Initialise containers
            objects = {}
            cbtn_param = []
            var_cbtn_param = []
            lbl_param = []
            ebox_param = []
            var_ebox_param = []
            unit_param = []
            # Radio button
            r_btn = tk.Radiobutton(master=frame, text=mod, variable=strvar_mod, value=mod, font=self.font_label, command=self.updateModelPL)
            objects['r_btn'] = r_btn
            r_btn.grid(row=idx+2, column=0, sticky='w')
            dict_mod[mod] = objects
            # Checkbuttons, labels, entryboxes and units
            for idx, param in enumerate(model.parameters[mod]):
                # Checkbutton
                intvar_check = tk.BooleanVar()
                intvar_check.set(True)
                cbtn = ttk.Checkbutton(master=frame, variable=intvar_check, command=self.checkFitFixedPL)
                cbtn_param.append(cbtn)
                var_cbtn_param.append(intvar_check)
                # Label
                lbl = tk.Label(master=frame, text=param.s_label+': ', font=self.font_label)
                lbl_param.append(lbl)
                # Entrybox
                strvar_ebox = tk.StringVar()
                ebox = tk.Entry(master=frame, textvariable=strvar_ebox, width=self.ebox_width)
                ebox_param.append(ebox)
                var_ebox_param.append(strvar_ebox)
                # Unit
                unit = tk.Label(master=frame, text=param.s_unit, font=self.font_label)
                unit_param.append(unit)
            objects['cbtn'] = cbtn_param
            objects['var_cbtn'] = var_cbtn_param
            objects['lbl'] = lbl_param
            objects['ebox'] = ebox_param
            objects['var_ebox'] = var_ebox_param
            objects['unit'] = unit_param
        return dict_mod

    def updateModelPL(self):
        # Update display
        # Remove present widgets
        for idx in range(len(self.models_PL.parameters[self.models_PL.active_model])):
            for el in ['cbtn', 'lbl', 'ebox', 'unit']:
                self.objects_PL[self.models_PL.active_model][el][idx].grid_forget()
        # Display new widgets
        self.models_PL.active_model = self.objects_PL['active'].get()
        for idx in range(len(self.models_PL.parameters[self.models_PL.active_model])):
            self.objects_PL[self.models_PL.active_model]['lbl'][idx].grid(row=2+idx, column=1, sticky='e')
            self.objects_PL[self.models_PL.active_model]['ebox'][idx].grid(row=2+idx, column=2)
            self.objects_PL[self.models_PL.active_model]['unit'][idx].grid(row=2+idx, column=3, sticky='w')
            self.objects_PL[self.models_PL.active_model]['cbtn'][idx].grid(row=2+idx, column=4, sticky='e')
        self.checkFitFixedPL()

    def checkFitFixedPL(self):
        for var, ebox in zip(self.objects_PL[self.models_PL.active_model]['var_cbtn'], self.objects_PL[self.models_PL.active_model]['ebox']):
            if var.get() == 0:
                ebox.config(state='normal')
            else:
                ebox.delete(0, 'end')
                ebox.config(state='disabled')
        all_selected = 1
        for var in self.objects['IF'][self.models['IF'].active_model]['var_cbtn']:
            all_selected *= var.get()
        for var in self.objects_PL[self.models_PL.active_model]['var_cbtn']:
            all_selected *= var.get()
        if all_selected == 0:
            self.objects['fit']['var_sel_all'].set(0)
        else:
            self.objects['fit']['var_sel_all'].set(1)

    def selectAllFit(self):
        if self.objects['fit']['var_sel_all'].get() == 1:
            set_val = 1
        else:
            set_val = 0
        for var in self.objects['IF'][self.models['IF'].active_model]['var_cbtn']:
            var.set(set_val)
        for var in self.objects_PL[self.models_PL.active_model]['var_cbtn']:
            var.set(set_val)
        self.checkFitFixedPL()

    def createFitObjects(self, frame, title):
        frame.columnconfigure(0, minsize=60)
        frame.columnconfigure(1, minsize=100)
        # Title
        (tk.Label(master=frame, text=title, font=self.font_title)
         .grid(row=0, column=0, sticky='w'))
        btn_fitIFPL = tk.Button(master=frame, text='Fit IF & PL', font=self.font_label, command=self.fitIF_PL)
        btn_fitIFPL.grid(row=1, column=0, sticky='w', pady=5)
        # Log plot
        var_log = tk.BooleanVar()
        cbtn_log = ttk.Checkbutton(master=frame, variable=var_log, command=self.setScalePlot, text='Log scale')
        cbtn_log.grid(row=2, column=0, columnspan=2, sticky='w', pady=5)
        # Select all
        var_select_all = tk.BooleanVar()
        cbtn_select_all = ttk.Checkbutton(master=frame, variable=var_select_all, command=self.selectAllFit, text='Select all')
        cbtn_select_all.grid(row=3, column=0, columnspan=2, sticky='w', pady=5)
        objects = {'btnIF_PL': btn_fitIFPL, 'var_sel_all': var_select_all, 'var_log': var_log}
        return objects

    def createOptionObjects(self, frame, title):
        frame.columnconfigure(2, minsize=150)
        # Title
        (tk.Label(master=frame, text=title, font=self.font_title)
         .grid(row=0, column=2, sticky='w'))
        (tk.Label(master=frame, text='x-axis variable', font=self.font_legend)
         .grid(row=1, column=2, sticky='w', pady=5, padx=5))
        strvar_x = tk.StringVar()
        strvar_x.set('w')
        tk.Radiobutton(master=frame, text='wavelength', variable=strvar_x, value='w', font=self.font_label, command=self.updateAxisUnit).grid(row=2, column=2, sticky='w')
        tk.Radiobutton(master=frame, text='energy', variable=strvar_x, value='e', font=self.font_label, command=self.updateAxisUnit).grid(row=3, column=2, sticky='w')
        objects = {'var_x': strvar_x}
        return objects

    def setScalePlot(self):
        if self.objects['fit']['var_log'].get() == 1:
            self.axes[1].set_yscale('log')
            self.axes[1].set_ylim(1e-4, 2)
        else:
            self.axes[1].set_yscale('linear')
            self.axes[1].set_ylim(-.05, 1.05)
        self.updateGraph()


    def createPlot(self):
        # Make figures
        figure = Figure(figsize=(5, 5))
        ax_top = figure.add_subplot(211)
        ax_top.xaxis.set_ticklabels([])
        ax_bot = figure.add_subplot(212)
        canvas = FigureCanvasTkAgg(figure, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, rowspan=5, sticky='news')

        toolbar = NavigationToolbar2Tk(canvas=canvas, window=self.frm_bar)
        toolbar.update()
        canvas._tkcanvas.grid()
        # Plot
        min_x = np.inf
        max_x = -np.inf
        IF_graph, = ax_top.plot(self.x_data['w']*self.x_scale['w'], self.models['IF'].model_class.IF, label='IF', linewidth=1.2, color='k')
        data_graphs = {}
        data_fix_graphs = {}
        PL_graphs = {}
        for idx, tab in enumerate(self.data_tabs):
            data_graph, = ax_top.plot(self.data_tabs[tab]['GUI'].data['w'][0]*self.data_tabs[tab]['GUI'].x_scale['w'], self.data_tabs[tab]['GUI'].data['w'][1],
                                      label=tab, linewidth=1.2, color=self.colors[idx % len(self.colors)])
            corrected_PL = self.data_tabs[tab]['GUI'].data['w'][1]/self.models['IF'].model_class.IF
            corrected_PL /= np.max(corrected_PL)
            BGF_w = E2W(self.data_tabs[tab]['GUI'].data['e'][0], self.data_tabs[tab]['GUI'].models['PL'].model_class.emission)[1]
            BGF_w /= np.max(BGF_w)
            data_fix_graph, = ax_bot.plot(self.data_tabs[tab]['GUI'].data['w'][0]*self.data_tabs[tab]['GUI'].x_scale['w'], corrected_PL,
                                          label=tab, linewidth=1.2, color=self.colors[idx % len(self.colors)])
            PL_graph, = ax_bot.plot(self.data_tabs[tab]['GUI'].data['w'][0]*self.x_scale['w'], BGF_w,
                                    linestyle='dotted', linewidth=1.2, color=self.colors[idx % len(self.colors)])
            min_x = min(min_x, np.min(self.data_tabs[tab]['GUI'].data['w'][0])*self.data_tabs[tab]['GUI'].x_scale['w'])
            max_x = max(max_x, np.max(self.data_tabs[tab]['GUI'].data['w'][0])*self.data_tabs[tab]['GUI'].x_scale['w'])
            data_graphs[tab] = data_graph
            data_fix_graphs[tab] = data_fix_graph
            PL_graphs[tab] = PL_graph


        ax_top.set_position([0.13, 0.55, 0.8, 0.42])
        ax_bot.set_position([0.13, 0.10, 0.8, 0.42])
        ax_bot.set_xlabel('Wavelength [μm]')
        for ax in [ax_top, ax_bot]:
            ax.legend(prop={'size': 9})
            ax.set_ylabel('Normalised PL [-]')
            ax.set_ylim(-.05, 1.05)
            ax.set_xlim(min_x, max_x)

        return figure, [ax_top, ax_bot], [data_graphs, {'IF': IF_graph}, data_fix_graphs, PL_graphs], toolbar

    def fitInit(self, label, tab=None):
        if label == 'IF':
            data = self
        elif label == 'PL':
            data = tab
        vars_bool = [var.get() for var in data.objects[label][data.models[label].active_model]['var_cbtn']]
        fixed_bool = [not val for val in vars_bool]
        vars = list(compress(data.models[label].parameters[data.models[label].active_model], vars_bool))
        vars_lb = [param.bounds[0] for param in vars]
        vars_ub = [param.bounds[1] for param in vars]
        vars_bounds = [vars_lb, vars_ub]
        vars_idx = np.cumsum(np.array(vars_bool).astype(int))-1
        params_values = list(data.models[label].model_class.params)
        vars_values = list(np.array(params_values)[vars_bool])
        vars_values = [vars_lb[idx] if vars_values[idx] < vars_lb[idx] else vars_values[idx] for idx in range(len(vars_values))]
        vars_values = [vars_ub[idx] if vars_values[idx] > vars_ub[idx] else vars_values[idx] for idx in range(len(vars_values))]
        return params_values, vars_values, vars_bounds, fixed_bool, vars_bool, vars_idx

    def setIFPLModels(self):
        for tab in self.data_tabs:
            # Set IF
            mod = self.models['IF'].active_model
            self.data_tabs[tab]['GUI'].objects['IF'][mod]['r_btn'].invoke()
            # Set PL
            mod = self.models_PL.active_model
            self.data_tabs[tab]['GUI'].objects['PL'][mod]['r_btn'].invoke()
            for idx, (check, ebox) in enumerate(zip(self.objects_PL[mod]['var_cbtn'], self.objects_PL[mod]['ebox'])):
                self.data_tabs[tab]['GUI'].objects['PL'][mod]['var_cbtn'][idx].set(check.get())
                if check.get() == 0:
                    if not ebox.get():
                        message = 'Please set value of fixed parameter.'
                        tk.messagebox.showwarning('Undefined parameter', message)
                        return
                    else:
                        self.data_tabs[tab]['GUI'].objects['PL'][mod]['var_sld'][idx].set(ebox.get())
            self.data_tabs[tab]['GUI'].updateParamFunction('PL')

    def fitIF_PL(self):
        self.setIFPLModels()
        params, vars_values, vars_bounds, fixed_bool, vars_bool, vars_idx = self.fitInit('IF')
        break_p = [len(params)]
        for tab in self.data_tabs:
            params_PL, vars_values_PL, vars_bounds_PL, fixed_bool_PL, vars_bool_PL, vars_idx_PL = self.fitInit('PL', self.data_tabs[tab]['GUI'])
            params += params_PL
            vars_values += vars_values_PL
            vars_bounds = [vars_bounds[0]+vars_bounds_PL[0], vars_bounds[1]+vars_bounds_PL[1]]
            fixed_bool += fixed_bool_PL
            vars_bool += vars_bool_PL
            vars_idx = np.concatenate((vars_idx, vars_idx_PL+1+vars_idx[-1]))
            break_p.append(break_p[-1]+len(params_PL))
        fit = curve_fit(lambda energy, *args: self.fitFunctionIF_PL(energy, break_p, *[params[idx] if fixed else args[not_fixed] for (idx, (fixed, not_fixed)) in enumerate(zip(fixed_bool, vars_idx))]), self.x_data['e'], np.zeros(self.x_data['e'].shape),
                        p0=vars_values, bounds=vars_bounds)
        new_params = np.array(params)
        new_params[vars_bool] = fit[0]
        for idx, (slider, param) in enumerate(zip(self.objects['IF'][self.models['IF'].active_model]['sld'], self.models['IF'].parameters[self.models['IF'].active_model])):
            slider.set(new_params[idx]/param.s_scale)
        self.models['IF'].model_class.params = new_params[:break_p[0]]
        for idx, tab in enumerate(self.data_tabs):
            data = self.data_tabs[tab]['GUI']
            for idx1, (slider, param) in enumerate(zip(data.objects['IF'][data.models['IF'].active_model]['sld'], data.models['IF'].parameters[data.models['IF'].active_model])):
                slider.set(new_params[idx1]/param.s_scale)
            data.models['IF'].model_class.params = new_params[:break_p[0]]
            for idx2, (slider, param) in enumerate(zip(data.objects['PL'][data.models['PL'].active_model]['sld'], data.models['PL'].parameters[data.models['PL'].active_model])):
                slider.set(new_params[break_p[idx]+idx2]/param.s_scale)
            data.models['PL'].model_class.params = new_params[break_p[idx]:break_p[idx+1]]
            data.updateGraph()
        self.updateGraph()

    def fitFunctionIF_PL(self, E, break_p, *params):
        self.models['IF'].model_class.params = list(params[0:break_p[0]])
        self.models['IF'].model_class.solve()
        diff = np.zeros(E.shape)
        for idx, tab in enumerate(self.data_tabs):
            PL_model = self.data_tabs[tab]['GUI'].models['PL'].model_class.fitFunction(E, *params[break_p[idx]:break_p[idx+1]])
            PL_w = E2W(E, PL_model)[1]
            y_model = PL_w*self.models['IF'].model_class.IF
            y_model /= np.max(y_model)
            diff += np.abs(y_model-self.data_tabs[tab]['GUI'].data['w'][1])
        return diff

    def updateAxisUnit(self):
        label = self.objects['options']['var_x'].get()
        if ((label == 'w' and self.current_axis == -1) or
           (label == 'e' and self.current_axis == 1)):
            self.current_axis *= -1
            self.updateGraph()
            for graphs in self.graphs:
                for name in graphs:
                    graphs[name].set_xdata(self.x_data[label]*self.x_scale[label])
            self.axes[0].set_xlim(np.min(self.x_data[label]*self.x_scale[label]), np.max(self.x_data[label]*self.x_scale[label]))
            self.axes[1].set_xlim(np.min(self.x_data[label]*self.x_scale[label]), np.max(self.x_data[label]*self.x_scale[label]))
            if label == 'w':
                for tab in self.data_tabs:
                    self.graphs[0][tab].set_ydata(self.data_tabs[tab]['GUI'].data['w'][1])
                    self.axes[1].set_xlabel('Wavelength [μm]')
            elif label == 'e':
                for tab in self.data_tabs:
                    self.graphs[0][tab].set_ydata(self.data_tabs[tab]['GUI'].data['e'][1])
                    self.axes[1].set_xlabel('Energy [eV]')

    def updateGraph(self):
        if self.models['IF'].active_model == 'off':
            self.models['IF'].model_class.IF = np.ones(self.models['IF'].model_class.IF.shape)
        else:
            self.models['IF'].model_class.solve()
        IF = self.models['IF'].model_class.IF
        if self.current_axis == 1:
            self.graphs[1]['IF'].set_ydata(IF)
        elif self.current_axis == -1:
            IF_e = W2E(self.models['IF'].model_class.wavelengths, IF)[1]
            self.graphs[1]['IF'].set_ydata(IF_e/np.max(IF_e))

        for tab in self.data_tabs:
            GUI = self.data_tabs[tab]['GUI']
            corrected_PL = GUI.data['w'][1]/self.models['IF'].model_class.IF
            PL_model = GUI.models['PL'].model_class.emission
            if self.current_axis == 1:
                self.graphs[2][tab].set_ydata(corrected_PL/np.max(corrected_PL))
                PL_w = E2W(GUI.models['PL'].model_class.E, PL_model)[1]
                self.graphs[3][tab].set_ydata(PL_w/np.max(PL_w))
            if self.current_axis == -1:
                corrected_PL_e = W2E(self.models['IF'].model_class.wavelengths,
                                     corrected_PL)[1]
                self.graphs[2][tab].set_ydata(corrected_PL_e/np.max(corrected_PL_e))
                self.graphs[3][tab].set_ydata(PL_model/np.max(PL_model))
        self.figure.canvas.draw_idle()

def W2E(wavelength, signal_wave):
    energy = h_eV*const.c/wavelength
    signal_e = signal_wave*wavelength**2/(h_eV*const.c)
    return np.flip(energy), np.flip(signal_e)


def E2W(energy, signal_e):
    wavelength = h_eV*const.c/energy
    signal_wave = signal_e*energy**2/(h_eV*const.c)
    return np.flip(wavelength), np.flip(signal_wave)


def main():
    # Create main window
    window = tk.Tk()
    window.geometry("969x715")
    window.title('PL Modeling Program')
    window.iconbitmap(path.join(path_to_file, 'files', 'logo.ico'))
    tabs = NotebookTabs(window)
    tk.mainloop()


if __name__ == "__main__":
    main()
