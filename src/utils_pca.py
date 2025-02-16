import scipy 
from scipy.sparse.linalg import eigs
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt 
from netCDF4 import Dataset

def rmnanersst(data):
    """
    Remove missing values (values <= -998) from the data.
    
    Args:
        data: Input data array (time x space)
    
    Returns:
        sst: Data with missing values removed
        miss: Indices of missing values in first time step
    """
    # Find missing values in first time step
    miss = np.where(data[0, :] <= -998)[0]
    
    # Get dimensions
    nrow, ncol = data.shape
    nmiss = len(miss)
    
    # Initialize output array
    sst = np.zeros((nrow, ncol - nmiss))
    
    # Copy valid data
    l = 0
    for i in range(ncol):
        if data[0, i] > -998:
            sst[:, l] = data[:, i]
            l += 1
    
    return sst, miss


def gsubsetindersst(data, extrop, box):
    """
    Extract SST data for specific ocean regions.
    
    Args:
        data: numpy array of SST data (time x space)
        extrop: integer flag for grid size (1 for ny=45, otherwise ny=13)
        box: string indicating region ('NINO3', 'NINO34', 'CEI', 'TNA', 'TSA')
    
    Returns:
        subdata: subset of data
        indsst: indices for the specified region
    """
    nrow, ncol0 = data.shape
    
    # Define grid dimensions
    nx = 180
    ny = 45 if extrop == 1 else 13
    
    # Initialize subdata array
    subdata = np.zeros((nrow, ny * nx))
    
    # Fill subdata array
    l = 0
    for j in range(ny):
        for i in range(nx):
            l = l + 1
            subdata[:, l-1] = data[:, l-1]
    
    # Find missing values
    miss = np.where(subdata[0, :] <= -998)[0]
    nmiss = len(miss)
    ncol = ny * nx
    
    # Initialize coordinate arrays
    x = np.zeros(ncol)
    y = np.zeros(ncol)
    xm = np.zeros(ncol - nmiss)
    ym = np.zeros(ncol - nmiss)
    
    # Grid spacing
    d = 2
    
    # Create coordinate grids
    m = 0
    for j in range(ny):
        for i in range(nx):
            m = m + 1
            x[m-1] = 0 + (i) * d
            y[m-1] = -27 + (j) * d
    
    # Remove missing values from coordinates
    l = 0
    for i in range(ncol):
        if subdata[0, i] > -999:
            l = l + 1
            xm[l-1] = x[i]
            ym[l-1] = y[i]
    
    # Find indices for specified region
    indsst = []
    for i in range(len(xm)):
        # NINO3 region (150W-90W)
        if box == 'NINO3' and 210 <= xm[i] <= 270 and -6 <= ym[i] <= 6:
            indsst.append(i)
        # NINO3.4 region (170W-120W)
        elif box == 'NINO34' and 190 <= xm[i] <= 240 and -6 <= ym[i] <= 6:
            indsst.append(i)
        # CEI region
        elif box == 'CEI' and 50 <= xm[i] <= 80 and -15 <= ym[i] <= 0:
            indsst.append(i)
        # TNA region
        elif box == 'TNA' and 320 <= xm[i] <= 340 and 5 <= ym[i] <= 20:
            indsst.append(i)
        # TSA region
        elif box == 'TSA' and 345 <= xm[i] <= 360 and -15 <= ym[i] <= -5:
            indsst.append(i)
    
    return subdata, np.array(indsst)


def gfiltcvl3ersst(data, npc, st1):
    """
    Perform EOF analysis on SST data.
    
    Args:
        data: numpy array of SST data
        npc: number of principal components
        st1: split point for the time series
    
    Returns:
        pcs: principal components
        eofs: empirical orthogonal functions
        sstmean1: mean of first period
        sstmean2: mean of second period
        sst: processed SST data
        teofs: reshaped EOFs
    """
    miss = np.where(data[0, :] <= -998)[0]
    nmiss = len(miss)
    nrow, ncol = data.shape
    
    # Remove missing values
    valid_points = data[0, :] > -998
    sst = data[:, valid_points]
    
    if st1 > 0:
        sstmean1 = np.mean(sst[:st1, :], axis=0)
        sstmean2 = np.mean(sst[st1:, :], axis=0)
        
        # Detrend data
        sstm1 = sst[:st1, :] - sstmean1
        sstm2 = sst[st1:, :] - sstmean2
        sstm = np.vstack((sstm1, sstm2))
    else:
        sstmean1 = np.mean(sst, axis=0)
        sstmean2 = sstmean1
        sstm = sst - sstmean1
    
    # Perform PCA
    U, S, Vh = linalg.svd(sstm, full_matrices=False)
    pcs = U[:, :npc] * S[:npc]
    eofs = Vh[:npc, :].T
    
    # Reshape EOFs
    nx, ny = 180, 45
    teofs = np.full((ny, nx, npc), np.nan)
    
    # Map EOFs back to spatial grid
    idx = 0
    for j in range(ny):
        for i in range(nx):
            if data[0, j*nx + i] > -998:
                teofs[j, i, :] = eofs[idx, :]
                idx += 1
    
    return pcs, eofs, sstmean1, sstmean2, sst, teofs


def compute_anomaly(data):
    """Equivalent to anomaly.m"""
    length = data.shape[0]
    
    # Center the data
    data = data - np.mean(data, axis=0)
    
    # Initialize arrays
    ndata = np.zeros_like(data)
    season = np.zeros((12, data.shape[1]))
    num = np.zeros(12)
    
    # Calculate seasonal cycle
    for l in range(length):
        month_idx = l % 12
        num[month_idx] += 1
        season[month_idx, :] += data[l, :]
    
    # Compute seasonal means
    season = season / num[:, np.newaxis]
    
    # Remove seasonal cycle
    for l in range(length):
        month_idx = l % 12
        ndata[l, :] = data[l, :] - season[month_idx, :]
    
    return ndata, num, season


def compute_pca(data, n_components):
    """Equivalent to pcas.m"""
    length, ncol = data.shape
    
    # Compute covariance matrix
    covmat = (data.T @ data) / length
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigs(covmat, k=n_components, which='LM')
    
    # Convert to real numbers (eigs returns complex numbers)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Compute principal components
    pcs = data @ eigenvectors
    
    return pcs, eigenvectors, eigenvalues


def ninom(pcs, eofs, indsst, sstmean):
    """
    Reconstruct SST anomalies from PC/EOF decomposition for a specific region.
    Exact implementation of MATLAB's ninom.m
    
    Args:
        pcs: Principal components (time x modes)
        eofs: Empirical Orthogonal Functions (space x modes)
        indsst: Indices for the region of interest
        sstmean: Mean SST field
    
    Returns:
        sstn: Reconstructed SST anomalies
    """
    length, nmax = pcs.shape
    nsst = len(indsst)
    sstn = np.zeros(length)
    
    # Implement exactly as MATLAB, keeping the same loop structure
    for l in range(length):
        for k in range(nsst):
            sstn[l] += sstmean[indsst[k]]
            for j in range(nmax):
                sstn[l] += pcs[l,j] * eofs[indsst[k],j]
    
    sstn = sstn / nsst
    return sstn

def pcas(data, npc):
    """
    Compute Principal Component Analysis (PCA) for the given data.
    Matches MATLAB's eigs behavior exactly.
    
    Args:
        data: Input data matrix (time x space)
        npc: Number of principal components to compute
    
    Returns:
        pcs: Principal components
        eofs: Empirical Orthogonal Functions
        d: Eigenvalues
    """
    length, ncol = data.shape
    
    # Compute covariance matrix exactly as MATLAB
    covmat = np.dot(data.T, data) / length
    
    # Use scipy's eigs with same parameters as MATLAB
    d, eofs = scipy.sparse.linalg.eigs(covmat, k=npc, which='LM', tol=0, maxiter=None)
    
    # Convert to real numbers (MATLAB automatically takes real part)
    eofs = np.real(eofs)
    d = np.real(d)
    
    # Compute PCs exactly as MATLAB
    pcs = np.dot(data, eofs)
    
    return pcs, eofs, d

def center(data):
    """
    Center the data by removing the mean.
    
    Args:
        data: Input data array (can be 2D or 3D)
        
    Returns:
        centered_data: Data with mean removed
    """
    # Handle both 2D and 3D arrays
    if len(data.shape) == 3:
        # For 3D array, compute mean along first dimension (time)
        mean_data = np.mean(data, axis=0)
        # Broadcast subtraction across time dimension
        centered_data = data - mean_data[np.newaxis, :, :]
    else:
        # For 2D array, compute mean along first dimension
        mean_data = np.mean(data, axis=0)
        # Subtract mean from each time point
        centered_data = data - mean_data
        
    return centered_data

def anomaly(data):
    """
    Calculate anomalies by removing seasonal cycle.
    
    Args:
        data: Input data array (time x space), 2D array
    
    Returns:
        ndata: Anomaly data
        num: Number of samples per month
        season: Seasonal cycle
    """
    # Get dimensions
    length, ny = data.shape
    nx = 1  # Always 1 as per MATLAB version
    
    # Remove mean first (as per MATLAB version)
    data = data - np.mean(data, axis=0)
    
    # Initialize arrays
    ndata = np.zeros((length, ny))
    season = np.zeros((12, ny))
    num = np.zeros(12)
    
    # Calculate seasonal cycle
    i = 0
    for l in range(length):
        i = i + 1
        if i > 12:
            i = 1
        num[i-1] = num[i-1] + 1
        season[i-1, :] = season[i-1, :] + data[l, :]
    
    # Normalize seasonal cycle
    for i in range(12):
        season[i, :] = season[i, :] / num[i]
    
    # Remove seasonal cycle
    i = 0
    for l in range(length):
        i = i + 1
        if i > 12:
            i = 1
        ndata[l, :] = data[l, :] - season[i-1, :]
    
    return ndata, num, season


# to be used by model evaluation
# take only the last T time stamps to agree with the prediction length 
# use only the post 317 ENSO modes to reconstruct the time series from PCs 
def reconstruct_enso(pcs, real_pcs, top_n_pcs=20, flag="test"):
    """
    Only use actual PCs (excluding sin/cos) for reconstruction
    """
    # If pcs has more features than real_pcs, assume the extra ones are sin/cos
    n_actual_pcs = min(top_n_pcs, real_pcs.shape[-1])
    pcs = pcs[..., :n_actual_pcs]
    real_pcs = real_pcs[..., :n_actual_pcs]
    
    pcs = np.squeeze(pcs)
    data = np.load("../data/pc_metadata.npz")
    if real_pcs is None:
        actual_pcs = data["pcs"]
    else:
        actual_pcs = real_pcs
        actual_pcs = np.squeeze(actual_pcs)   
    assert pcs.shape == actual_pcs.shape
    eofs = data["eofs"]
    sstmean1 = data["sstmean1"]  
    sstmean2 = data["sstmean2"]
    indsst = data["indsst"]
    # split_year = data["split_year"]
    # nino34_20_1 = ninom(actual_pcs[:split_year, :20], eofs[:, :20], indsst, sstmean1)
    # nino34_20_1_pred = ninom(pcs[:split_year, :20], eofs[:,:20], indsst, sstmean1)
    sstmean = sstmean1 if flag == "train" else sstmean2
    nino34 = ninom(actual_pcs[:, :top_n_pcs], eofs[:, :top_n_pcs], indsst, sstmean)
    nino34_pred = ninom(pcs[:, :top_n_pcs], eofs[:,:top_n_pcs], indsst, sstmean)
    # nino34 = np.concatenate([nino34_20_1, nino34_20_2])
    # print(f"nino34 = {nino34.shape}, nino34_pred = {nino34_pred.shape}")
    return nino34, nino34_pred

if __name__ == "__main__":
    ds = Dataset('../data/ersst2024-12.nc')
    lon = ds.variables['X'][:]
    lat = ds.variables['Y'][:]
    anom = ds.variables['anom'][:].squeeze()
    anom = np.array(anom)

    ssta = anom
    rawdata = ssta.reshape(ssta.shape[0],  ssta.shape[1] * ssta.shape[2])
    rawdata = anomaly(rawdata)[0] 
    tmp, indsst = gsubsetindersst(rawdata, extrop=1, box='NINO34')
    tdata, _ = rmnanersst(tmp)
    nino34 = np.mean(tdata[:, indsst], axis=1)
    split_year = 317  # Corresponds to specific year split point
    pcs, eofs, sstmean1, sstmean2, sst, teofs = gfiltcvl3ersst(rawdata, npc=20, st1=split_year)

    # temporary, save the 1st 20 pcs and 1st 20 eofs, also save indsst and sstmeans for later reconstruction in model evaluation
    with open("../data/pc_metadata.npz", "wb") as f:
        np.savez(f, pcs=pcs, eofs=eofs, sstmean1=sstmean1, sstmean2=sstmean2, indsst=indsst, split_year=split_year)

    # Calculate NINO3.4 reconstruction with 5 PCs
    nino34_5_1 = ninom(pcs[:split_year, :5], eofs[:, :5], indsst, sstmean1)
    nino34_5_2 = ninom(pcs[split_year:, :5], eofs[:, :5], indsst, sstmean2)
    nino34_5 = np.concatenate([nino34_5_1, nino34_5_2])

    # Calculate NINO3.4 reconstruction with 20 PCs
    nino34_20_1 = ninom(pcs[:split_year, :20], eofs[:, :20], indsst, sstmean1)
    nino34_20_2 = ninom(pcs[split_year:, :20], eofs[:, :20], indsst, sstmean2)
    nino34_20 = np.concatenate([nino34_20_1, nino34_20_2]) 

    month = np.arange(1, len(pcs) + 1)
    year = 1950 + month/12

    plt.figure(figsize=(12, 6))
    plt.plot(year, nino34, 'k', label='Full field')
    plt.plot(year, nino34_5, 'r', label='5 PCs')
    plt.plot(year, nino34_20, 'b', label='20 PCs')

    plt.legend()
    plt.xlim(year[0], year[-1])
    plt.grid(True)
    plt.title('NINO3.4 Index')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (°C)')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save plot (optional)
    plt.savefig('nino34_index.png', dpi=300, bbox_inches='tight')

