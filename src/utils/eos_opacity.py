import numpy as np

def read_propaceos( file_name, 
                need_zf_table=False, 
                need_en_table=False, 
                need_eion=False, 
                need_eele=False, 
                need_pion=False, 
                need_pele=False,
                need_ross_opacity=False, 
                need_emiss_opacity=False, 
                need_abs_opacity=False ):   

    """
    Reads tabulated data from a PROPACEOS file.

    Args:
        file_name (str): Path to the PROPACEOS data file.
        need_zf_table (bool): Whether to read average ionization data.
        need_en_table (bool): Whether to read internal energy data.
        need_hc_table (bool): Whether to read specific heat data.

    Returns:
        dict: A dictionary with extracted data. Keys depend on requested tables.
    """

    # Conversion factors
    eV_to_K = 11604.5221
    joule_to_erg = 1.0e7
    temperatures = []
    densities    = []
    groups       = []
    # Initialize output
    data = {
        "temperatures": None,
        "densities": None,
        "rad_groups": None,
        "zf_table": None,
        "ross_opacity": None,
        "emiss_opacity": None,
        "abs_opacity": None,
        "en_table": None,
        "eion_table": None,
        "eele_table": None,
        "pion_table": None,
        "pele_table": None,
    }

    with open(file_name, "r") as file:
        # Skip header lines
        for _ in range(38):
            next(file)

        # Read number of temperature steps and temperatures
        nsteps_temperature = int(file.readline().strip())
        if nsteps_temperature <= 0:
            raise ValueError("No temperature grid found in the PROPACEOS file.")
        
        for _ in range(int(nsteps_temperature//10)):
            temperatures.extend([float(x) for x in file.readline().strip().split()])

        # Read number of density steps and densities
        nsteps_density = int(file.readline().strip())
        if nsteps_density <= 0:
            raise ValueError("No density grid found in the PROPACEOS file.")

        for _ in range(int(nsteps_density//10)):
            densities.extend([float(x) for x in file.readline().strip().split()])

        # Store temperature and density grids
        data["temperatures"] = np.array(temperatures)   # eV
        data["densities"] = np.array(densities)         # cm^-3

        # Read grid for opacities (assume grid is skipped in the file)
        for _ in range(nsteps_temperature//10+nsteps_density//10+2+5):
            next(file)

        # Skip energy group grids
        ngroups_energy = int(file.readline().strip())
        next(file)  # Skip one line

        for _ in range(ngroups_energy//10 + 1):
            groups.extend([float(x) for x in file.readline().strip().split()])

        data["rad_groups"] = np.array(groups)

        # Read ZF table (if requested)
        if need_zf_table:
            next(file) # Skip a line
            zf_table = np.zeros((nsteps_temperature, nsteps_density))  # Initialize the 2D grid
            for t in range(nsteps_temperature):
                zf_row = []  # Temporary storage for one temperature's density values
                for _ in range(nsteps_density // 10):  # Read 10 values at a time for one temperature row
                    zf_row.extend([float(x) for x in file.readline().strip().split()])
                zf_table[t, :] = zf_row  # Assign the full density row to the table
            data["zf_table"] = zf_table

        # Read Rosseland mean opacity table (if requested)
        if need_ross_opacity:
            next(file)  # Skip a line
            ross_table = np.zeros((nsteps_temperature, nsteps_density))
            for t in range(nsteps_temperature):
                ross_row = []
                for _ in range(nsteps_density // 10):
                    ross_row.extend([float(x) for x in file.readline().strip().split()])
                ross_table[t, :] = ross_row
            data["ross_opacity"] = ross_table  # Units: cm^2/g

        # Read emission opacity table (if requested)
        if need_emiss_opacity:
            next(file)  # Skip a line
            emiss_table = np.zeros((nsteps_temperature, nsteps_density))
            for t in range(nsteps_temperature):
                emiss_row = []
                for _ in range(nsteps_density // 10):
                    emiss_row.extend([float(x) for x in file.readline().strip().split()])
                emiss_table[t, :] = emiss_row
            data["emiss_opacity"] = emiss_table  # Units: cm^2/g

        # Read absorption opacity table (if requested)
        if need_abs_opacity:
            next(file)  # Skip a line
            abs_table = np.zeros((nsteps_temperature, nsteps_density))
            for t in range(nsteps_temperature):
                abs_row = []
                for _ in range(nsteps_density // 10):
                    abs_row.extend([float(x) for x in file.readline().strip().split()])
                abs_table[t, :] = abs_row
            data["abs_opacity"] = abs_table  # Units: cm^2/g
        
        # Read EN table (if requested)
        if need_en_table:
            file.readline()  # Skip a line
            en_table = np.zeros((nsteps_temperature, nsteps_density))  # Initialize 2D grid

            for t in range(nsteps_temperature):
                en_row = []  # Temporary storage for one temperature's density values
                for _ in range(nsteps_density // 10):  # Read 10 values at a time for one temperature row
                    en_row.extend([float(x) for x in file.readline().strip().split()])
                en_table[t, :] = en_row  # Assign the full density row to the table
            data["en_table"] = en_table    # Units: Joule/g

        # Read ion energy table (if requested)
        if need_eion:
            next(file)  # Skip a line
            eion_table = np.zeros((nsteps_temperature, nsteps_density))
            for t in range(nsteps_temperature):
                eion_row = []
                for _ in range(nsteps_density // 10):
                    eion_row.extend([float(x) for x in file.readline().strip().split()])
                eion_table[t, :] = eion_row
            data["eion_table"] = eion_table    # Units: Joule/g

        # Read electron energy table (if requested)
        if need_eele:
            next(file)  # Skip a line
            eele_table = np.zeros((nsteps_temperature, nsteps_density))
            for t in range(nsteps_temperature):
                eele_row = []
                for _ in range(nsteps_density // 10):
                    eele_row.extend([float(x) for x in file.readline().strip().split()])
                eele_table[t, :] = eele_row
            data["eele_table"] = eele_table      # Units: Joule/g

        # Read ion pressure table (if requested)
        if need_pion:
            next(file)  # Skip a line
            pion_table = np.zeros((nsteps_temperature, nsteps_density))
            for t in range(nsteps_temperature):
                pion_row = []
                for _ in range(nsteps_density // 10):
                    pion_row.extend([float(x) for x in file.readline().strip().split()])
                pion_table[t, :] = pion_row
            data["pion_table"] = pion_table       # Units: dyne/cm**2

        # Read electron pressure table (if requested)
        if need_pele:
            next(file)  # Skip a line
            pele_table = np.zeros((nsteps_temperature, nsteps_density))
            for t in range(nsteps_temperature):
                pele_row = []
                for _ in range(nsteps_density // 10):
                    pele_row.extend([float(x) for x in file.readline().strip().split()])
                pele_table[t, :] = pele_row
            data["pele_table"] = pele_table     # Units: dyne/cm**2

    return data