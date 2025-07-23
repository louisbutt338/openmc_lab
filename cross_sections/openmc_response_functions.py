import openmc
import matplotlib.pyplot as plt
from matplotlib import rc
rc("font", **{"family":"sans-serif", "sans-serif":["Helvetica"]},weight='normal',size=20)
import numpy as np
import pandas as pd
import os
import actigamma as ag
import json

####################

# choose default data library: endfb8 or tendl21 or irdff2
xs_library = 'tendl21'

# where are the cross-sections?
xs_folder_path = '/Users/ljb841@student.bham.ac.uk/MCNP/MCNP_DATA'

# exciting opportunity to create your own group structure to read the cross-sections in 
logspace_gs = np.logspace(-2, 6, num=16, base=10,endpoint=False)
linspace_gs = np.linspace(1e6, 20e6, 100)
custom_gs = np.concatenate((logspace_gs,linspace_gs))

# load the vitamin-j group structures
vj175_gs = openmc.mgxs.EnergyGroups('VITAMIN-J-175')
vj211_gs = json.load(open(f'vitamin-j-211.json')).values()

#choose energy group between custom_gs, vj175_gs and vj211_gs
energy_group = vj175_gs #choose energy group

# gimme the name of the json file with all the foil data in and also the labels for the isotopes (in order) for plotting
data_file_name = 'pli_data'
reaction_labels = [r'${}^{115}$In(n,$\gamma$)',
                r'${}^{164}$Dy(n,$\gamma$)',
                r'${}^{197}$Au(n,$\gamma$)',
                r"${}^{115}$In(n,n')", 
                r'${}^{65}$Cu(n,p) *',
                r'${}^{56}$Fe(n,p)',
                r'${}^{27}$Al(n,$\alpha$)', 
                r'${}^{197}$Au(n,2n)',
                r'${}^{93}$Nb(n,2n)',
                r'${}^{58}$Ni(n,2n) ']
#data_file_name = 'dli_data'
#reaction_labels = [r'${}^{115}$In(n,$\gamma$)',  
#                r'${}^{164}$Dy(n,$\gamma$) *',
#                r'${}^{89}$Y(n,$\gamma$) *',  
#                r'${}^{197}$Au(n,$\gamma$)',  
#                r"${}^{115}$In(n,n')",       
#                r'${}^{58}$Ni(n,p)',         
#                r'${}^{27}$Al(n,p)',          
#                r'${}^{65}$Cu(n,p) *',       
#                r'${}^{56}$Fe(n,p)',         
#                r'${}^{27}$Al(n,$\\alpha$)',  
#                r'${}^{197}$Au(n,2n)',       
#                r'${}^{58}$Ni(n,np) *',       
#                r'${}^{45}$Sc(n,2n) *',       
#                r'${}^{58}$Ni(n,2n)',        
#                r'${}^{103}$Rh(n,3n) *']      

#whatever you want to name the figure that comes out
figure_name = f'plimar24_{xs_library}_{len(energy_group.group_edges)-1}_test'

####################

# extract xs information from IRDFF ACE file
def irdff2_xs_extraction(irdff_ace_filepath,mt_number,energy_bins):
    ace_table = openmc.data.ace.get_table(irdff_ace_filepath)
    nxs = ace_table.nxs
    jxs = ace_table.jxs
    xss = ace_table.xss
    lmt = jxs[3]
    nmt = nxs[4]
    lxs = jxs[6]
    mts = xss[lmt : lmt+nmt].astype(int)
    #print(mts)
    locators = xss[lxs : lxs+nmt].astype(int)
    cross_sections = {}
    for mt, loca in zip(mts, locators):
    # Determine starting index on energy grid
        nr = int(xss[jxs[7] + loca - 1])
        if nr == 0:
            breakpoints = None
            interpolation = None
        else:
            breakpoints = xss[jxs[7] + loca : jxs[7] + loca + nr].astype(int)
            interpolation = xss[jxs[7] + loca + nr : jxs[7] + loca + 2*nr].astype(int)
        # Determine number of energies in reaction
        ne = int(xss[jxs[7] + loca + 2*nr])
        # Read reaction cross section
        start = jxs[7] + loca + 1 + 2*nr
        energy = xss[start : start + ne] * 1e6
        xs = xss[start + ne : start + 2*ne]
        cross_sections[mt] = openmc.data.Tabulated1D(energy, xs, breakpoints, interpolation)
    return cross_sections[mt_number](energy_bins)

# little switch to get the right format to read to the TENDL ACE filenames
def tendl_extraction(isotope):
    isotope_tendl_format = isotope
    if len(isotope) == 4:
        isotope_tendl_format = isotope[:2] + '0' + isotope[2:]
    if len(isotope) == 3:
        isotope_tendl_format = isotope[:1] + '0' + isotope[1:]
    return isotope_tendl_format

#correction for self shielding of neutrons for the thermal reactions
def thermal_self_shielding(atom_density,cross_section,thickness):
    sigma_and_t = atom_density*cross_section*thickness
    #tau = atom_density*200*(0.025/0.4)
    #tau_integral = lambda t:  ( (1-np.exp(-t)) / t)
    #y,err = integrate.quad(tau_integral , 0, tau)
    #correction_factor = (1/tau) * y

    #correction_factor = 1 - (1/2)*atom_density*cross_section*thickness + (1/6)*(atom_density*cross_section*thickness)**2
    #correction_factor = 1 - (1/np.sqrt(np.pi))*(atom_density*cross_section[1]*thickness)*(np.sqrt(293.6/100)) + (1/3)*((atom_density*cross_section[1]*thickness)*np.sqrt(293.6/100))**2
    #correction_factor = 1 - (1/2)*sigma_and_t*(np.log(1/sigma_and_t)+(3/2)-0.5772156) - (1/6)*(sigma_and_t)**2
    correction_factor = (1-np.exp(-sigma_and_t)) / sigma_and_t 
    return correction_factor

def resonance_self_shielding(isotope):
    res_ssf_json = json.load(open(f'ssf_{len(energy_group.group_edges)-1}.json'))
    isotope_res_ssf_array = res_ssf_json[isotope][::-1]
    return isotope_res_ssf_array

# the big horrible information extractor from all libraries, extract RF and XS and atom density form the data - CHANGE SO EACH LIBRARY HAS OWN FUNCTION
def reaction_info(isotope, material_number, mt_number, density, mass,thickness):
    foil = ''.join(filter(str.isalpha, isotope))  
    ace_filename = str(material_number)[:5] + '.800nc'
    energy_bins = energy_group.group_edges
    material = openmc.Material()
    material.set_density('g/cm3', density)
    material.add_element(foil, 1) 
    isotope_atom_density = material.get_nuclide_atom_densities()[isotope]
    foil_volume = mass / material.density

    # for running majority endf isotopes
    if xs_library == 'endfb8':
        PATH_ACE = '{}/Lib80x/{}/{}'.format(xs_folder_path,foil,ace_filename)
        isotope_data = openmc.data.IncidentNeutron.from_ace(PATH_ACE)
        cross_section = (isotope_data[mt_number].xs['294K'](energy_bins))

    # for running majority tendl isotopes
    if xs_library == 'tendl21':
        if mt_number in [103] and isotope in ['Cu65']:
            PATH_ACE = '{}/Lib80x/{}/{}'.format(xs_folder_path,foil,ace_filename)
            isotope_data = openmc.data.IncidentNeutron.from_ace(PATH_ACE)
            cross_section = (isotope_data[mt_number].xs['294K'](energy_bins))
        if mt_number in [11004,11016] and isotope in ['In115','Nb93']:
            filename_irdff_format = ace_filename[:6] + '34y'
            PATH_ACE = '{}/IRDFF-II/endf_format/{}'.format(xs_folder_path,filename_irdff_format)
            cross_section = (irdff2_xs_extraction(PATH_ACE,mt_number,energy_bins))
        else:
            PATH_ACE = '{}/tendl21c/tendl21c/{}'.format(xs_folder_path,tendl_extraction(isotope))
            isotope_data = openmc.data.IncidentNeutron.from_ace(PATH_ACE)
            cross_section = (isotope_data[mt_number].xs['294K'](energy_bins))

    # for running majority irdff isotopes
    if xs_library == 'irdff2':
        if mt_number not in [28,104,17] and isotope not in ['Cu65', 'Dy164', 'Y89','Sc45','Rh103'] : 
            filename_irdff_format = ace_filename[:6] + '34y'
            PATH_ACE = '{}/IRDFF-II/endf_format/{}'.format(xs_folder_path,filename_irdff_format)
            cross_section = (irdff2_xs_extraction(PATH_ACE,mt_number,energy_bins))
        else:
            PATH_ACE = '{}/Lib80x/{}/{}'.format(xs_folder_path,foil,ace_filename)
            isotope_data = openmc.data.IncidentNeutron.from_ace(PATH_ACE)
            cross_section = (isotope_data[mt_number].xs['294K'](energy_bins))

    # calculate the response function and output the information
    if mt_number == 102:
        uncorrected_response_function = isotope_atom_density*foil_volume*cross_section
        response_function = (uncorrected_response_function
                             *thermal_self_shielding(isotope_atom_density,cross_section,thickness))
                             #*resonance_self_shielding(isotope))
    else:
        response_function = isotope_atom_density*foil_volume*cross_section
    return energy_bins,isotope_atom_density,foil_volume,cross_section,response_function

def plotter_and_dumper(data_dictionary):
    # output data into csv format and plot
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data_dictionary.keys()))))
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,8),gridspec_kw={'width_ratios': [2, 3.5]},tight_layout=True)
    for reaction in list(data_dictionary.keys())[0:]:
        c=next(color)
        ax1.step(data_dictionary[reaction]      [0]/1e6, data_dictionary[reaction]    [4], label= reaction,c=c)    
        ax2.step(data_dictionary[reaction]      [0]/1e6, data_dictionary[reaction]    [4], label= reaction,c=c)
        df_response_matrices = pd.DataFrame(data_dictionary[reaction]   [4][1:]).T
        df_response_matrices.to_csv(f'response_matrix_{len(energy_group.group_edges)-1}.csv',
                                    index=False,header=False,mode='a')
    df_reactions = pd.DataFrame(data_dictionary.keys())
    df_reactions.to_csv(f'reaction_rate_labels_{len(energy_group.group_edges)-1}.csv',
                        index=False,header=False,mode='w')
    df_energygroup = pd.DataFrame(energy_group.group_edges)
    df_energygroup.to_csv(f'group_structure_{len(energy_group.group_edges)-1}.csv',
                          index=False,header=False,mode='w')

    # plotting parameters
    ax1.set_xlim(1e-8,1e0)
    ax1.set_ylim(1e-11,1e3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid()
    ax2.set_xlim(1e0,18)
    ax2.set_ylim(1e-11,1e3)
    ax2.tick_params(axis='y',left=False,labelleft=False)
    ax2.set_yscale('log')
    ax2.grid()  
    ax2.legend(loc="upper right", frameon=True, fontsize=18,
               fancybox=False,facecolor='white',framealpha=1,ncol=3) #bbox_to_anchor=(0.025, 0.955)
    fig.supylabel('Response function Rn(E) (cm$^2$)',y=0.55)
    fig.supxlabel('Neutron energy (MeV)',y=0.03)
    plt.savefig(f'{figure_name}.png')

def run():
    # remove existing response matrix file
    if os.path.isfile(f'response_matrix_{len(energy_group.group_edges)-1}.csv') == True:
        os.remove(f'response_matrix_{len(energy_group.group_edges)-1}.csv')

    # load json data into the silly big dictionary method you used initially
    json_file_data = json.load(open(f'{data_file_name}.json'))
    material_list = [x['mat_number'] for x in json_file_data.values()]
    mt_list = [x['mt_value'] for x in json_file_data.values()]
    density_list = [x['density_gcm3'] for x in json_file_data.values()]
    mass_list = [x['mass_g'] for x in json_file_data.values()]
    thickness_list = [x['thickness_cm'] for x in json_file_data.values()]
    parent_isotopes_list = [x['parent_isotope'] for x in json_file_data.values()]
    foil_data_dictionary = {}

    # run the plotter and info dumper
    for i in range(len(reaction_labels)):
        foil_data_dictionary[reaction_labels[i]] = reaction_info(parent_isotopes_list[i],material_list[i],
                                                                 mt_list[i],density_list[i] ,mass_list[i],thickness_list[i])
    plotter_and_dumper(foil_data_dictionary)

run()