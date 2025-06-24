"""
Library to handle input and output filetypes.

Author: Louis Evans
Reviewer: Stefano Merlini
"""

import numpy as np
import pyvista as pv

def export_pvti(arr: np.ndarray, fname: str = None, extent_x = None, extent_y = None, extent_z = None):

    '''
    Export a 3d array as a pvti file format
        fname: str, file path and name to save under. A VTI pointed to by a PVTI file are saved in this location. If left blank, the name will default to:
                ./plasma_PVTI_DD_MM_YYYY_HR_MIN    
    '''

    if fname is None:
        import datetime as dt
        year = dt.datetime.now().year
        month = dt.datetime.now().month
        day = dt.datetime.now().day
        min = dt.datetime.now().minute
        hour = dt.datetime.now().hour

        fname = f'./plasma_PVTI_{day}_{month}_{year}_{hour}_{min}' #default fname to the current date and time 

    try: #check to ensure electron density has been added
        np.shape(arr)
        rnec = arr
    except:
        raise Exception('No electron density currently loaded!')

    # Create the spatial reference  
    grid = pv.ImageData()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    # the CELL data
    grid.dimensions = np.array(rnec.shape) + 1
    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set

    if extent_x is None:
        extent_x = (np.shape(arr)[0])//2
    if extent_y is None:
        extent_y = (np.shape(arr)[1])//2
    if extent_z is None:
        extent_z = (np.shape(arr)[2])//2

    xc,yc,zc = np.linspace(-extent_x,extent_x, arr.shape[0]), np.linspace(-extent_y,extent_y, arr.shape[1]), np.linspace(-extent_z,extent_z, arr.shape[2])

    #scaling
    x_size = np.max(xc) / ((np.shape(arr)[0])//2 )  #assuming centering about the origin
    y_size = np.max(yc) / ((np.shape(arr)[1])//2 ) 
    z_size = np.max(zc) / ((np.shape(arr)[2])//2 )
    grid.spacing = (x_size, y_size, z_size)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data["rnec"] = rnec.flatten(order="F")  # Flatten the array

    grid.save(f'{fname}.vti')

    print(f'VTI saved under {fname}.vti')

    #prep values to write the pvti, written to match the exported vti using pyvista
    relative_fname = fname.split('/')[-1]
    spacing_x = (2*xc.max())/np.shape(xc)[0]
    spacing_y = (2*yc.max())/np.shape(yc)[0]
    spacing_z = (2*zc.max())/np.shape(zc)[0]
    content = f'''<?xml version="1.0"?>
                    <VTKFile type="PImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
                    <PImageData WholeExtent="0 {np.shape(arr)[0]} 0 {np.shape(arr)[1]} 0 {np.shape(arr)[2]}" GhostLevel="0" Origin="0 0 0" Spacing="{x_size} {y_size} {z_size}">
                            <PCellData Scalars="rnec">
                                <PDataArray type="Float64" Name="rnec">
                                </PDataArray>
                            </PCellData>
                            <Piece Extent="0 {np.shape(arr)[0]} 0 {np.shape(arr)[1]} 0 {np.shape(arr)[2]}" Source="{relative_fname}.vti"/>
                    </PImageData>
                    </VTKFile>'''
    # write file
    with open(f'{fname}.pvti', 'w') as file:
        file.write(content)
    print(f'Scalar Domain electron density succesfully saved under {fname}.pvti !')

def pvti_readin(filename):
    '''
	Reads in data from pvti with filename, use this to read in electron number density data
	'''
    import vtk
    import os
    from vtk.util import numpy_support as vtk_np

    reader = vtk.vtkXMLPImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    dim = data.GetDimensions()
    spacing = np.array(data.GetSpacing())
    v = vtk_np.vtk_to_numpy(data.GetCellData().GetArray(0))
    n_comp = data.GetCellData().GetArray(0).GetNumberOfComponents()
    vec = [int(i-1) for i in dim]
    if(n_comp > 1):
        vec.append(n_comp)
    if(n_comp > 2):
        img = v.reshape(vec,order="F")[0:dim[0]-1,0:dim[1]-1,0:dim[2]-1,:]
    else:
        img = v.reshape(vec,order="F")[0:dim[0]-1,0:dim[1]-1,0:dim[2]-1]
    dim = img.shape
    return img, dim, spacing

def hdf_readin(filename):
    '''
    Read in hdf5 files and return electron density field
    '''
    import yt
    ds = yt.load(filename)

    def _ne(field, data):
        return (
            6.022e23
            * data["flash", "dens"]
            * data["flash", "ye"]   #ye is Z/A
        )

    ds.add_field(
        name=("flash","ne"),
        function=_ne,
        sampling_type="local",
        units="code_mass/code_length**3",
        force_override=False
    )

    level = ds.index.max_level
    dims = ds.domain_dimensions * ds.refine_by**level
    cube = ds.covering_grid(level, left_edge=ds.domain_left_edge, dims=dims)
    ne   = cube[("flash","ne")]
    spacing = [(ds.domain_right_edge[i] - ds.domain_left_edge[i]) / dims[i] for i in range(len(dims))]

    return ne, dims, spacing

def hdf_to_pvti(hdf_filename, pvti_filename):
    '''
    convert hdf5 format to pvti format
    '''
    ne, dims, spacing = hdf_readin(hdf_filename)
    extent_x = (dims[0]*spacing[0])/2
    extent_y = (dims[1]*spacing[1])/2
    extent_z = (dims[2]*spacing[2])/2
    
    export_pvti(ne, fname = pvti_filename, extent_x = extent_x, extent_y = extent_y, extent_z = extent_z)