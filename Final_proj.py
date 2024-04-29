#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:09:59 2024
The PDSI is surface based and the spatial extent
 is 2.5 degrees latitude by 2.5 degrees longitude global grid (144x55). 
 I can everything listed with PDSI and streamflow, yes. 


 
Is the PDSI output surface based then? What is the spatial extent?
 
Can you provide a time period that covers a climate event and show time series of streamflow,
 a temporal average of temperature and precip conditions (spatial plots)? 
 Something to that extent to highlight how your dataset can characterize that time period?
 
 
 Guesss i need to create a power point
 
@author: bam
"""
############
#Starting code
# Import the everything that I will need and will need 
import numpy as np
import netCDF4
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import glob
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

######################

#Import the PDSI
pdsi = netCDF4.Dataset('nada_hd2_cl.nc')
#Run in the CRU precip data
pr=xr.open_dataset('cru_ts4.07.1901.2022.pre.dat.nc').pre
#Take the mean of "time" to remove the variable to not be a time series
pr_clim=pr.mean('time')
#The data has the station number as its geographic identifier, so this converts it to lon lat for plotting
station_all=pd.read_csv('lat_lon_points',sep='\t',skiprows=26).iloc[1:,:]
#This is all my streamflow streams
in_file=glob.glob('/Users/bam/Documents/fimallanb/stream/*')
#Once again all this code changes the site numbers to lon lat
data={'Name':[],'Latitude':[],'Longitude':[]}
#This block uses the glob library to show all files in a directory, 
# In each file, it reads the file, extracts station ids, and looks up the corresponding lat lon.
for ifile in in_file:
    readin_file=pd.read_csv(ifile,delim_whitespace=True,skiprows=35).iloc[1:,:]
    st=readin_file['site_no'].iloc[0]
    st_idx=station_all.site_no.index[station_all.site_no==st].tolist()[0]
    st_lat=float(station_all.iloc[st_idx,3])
    st_lon=float(station_all.iloc[st_idx,4])
    data['Name'].append(st)
    data['Latitude'].append(st_lat)
    data['Longitude'].append(st_lon)
#Plots the figure, this sets up the size and what data wil be plotted
fig = plt.figure(figsize=(10, 3),dpi=250)
ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))    
plot_lev=np.arange(00,181,15)
plot_var=(pr_clim)
#Make the graphs look pretty 
ix1=ax.contourf(plot_var.lon,plot_var.lat,plot_var,levels=plot_lev,extend='both',cmap='jet_r',transform=ccrs.PlateCarree(),zorder=1,alpha=0.8)
ix2=ax.contourf(plot_var.lon,plot_var.lat,plot_var,levels=plot_lev,extend='neither',cmap='jet_r',transform=ccrs.PlateCarree(),zorder=1,alpha=0.8)        
ax.set_extent([220 ,310,20,55], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linewidth=0.5,zorder=10)
ax.add_feature(cfeature.STATES, linewidth=0.2,alpha=0.5,linestyle=':',zorder=11)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5,zorder=10)
cba = plt.colorbar(ix2,  extend='neither', orientation='vertical',pad=0,shrink=1.0, ax=ax)
cba.set_ticks(np.arange(0,181,30),minor=False)
cba.ax.tick_params(labelsize=8,direction='in',left='False',width=0.2,pad=0.3)
cba.set_label('mm',fontsize=6)
plt.title('CRU precip data')
plt.show()  

########### THIS PLOTTED THE CRU DATA ##########


# I will now outline the MRB
# THis reads in a shapefile of the MRB(Mississippi River Basin)
mrb = gpd.read_file('/Users/bam/Documents/fimallanb/Shapefile /Miss_RiverBasin.shp')

#This is a geodataframe that makes sure that there is lat lon to plot the outline effectivly 
mrb=mrb.to_crs(4326)

#These are once again the outline of the plots, and the plotted data
fig = plt.figure(figsize=(10, 3),dpi=250)
ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))    
plot_lev=np.arange(00,181,15)
plot_var=(pr_clim)
ix1=ax.contourf(plot_var.lon,plot_var.lat,plot_var,levels=plot_lev,extend='both',cmap='jet_r',transform=ccrs.PlateCarree(),zorder=1,alpha=0.8)
ix2=ax.contourf(plot_var.lon,plot_var.lat,plot_var,levels=plot_lev,extend='neither',cmap='jet_r',transform=ccrs.PlateCarree(),zorder=1,alpha=0.8)
feat1=cfeature.ShapelyFeature(mrb['geometry'],crs=ccrs.PlateCarree(),facecolor='None',edgecolor='k',linewidth=0.7,zorder=13)
#Feature 1 is the shapefile of the MRB
ax.add_feature(feat1)    
ax.set_extent([220 ,310,20,55], crs=ccrs.PlateCarree())
#Make the plots look pretty 
ax.add_feature(cfeature.BORDERS, linewidth=0.5,zorder=10)
ax.add_feature(cfeature.STATES, linewidth=0.2,alpha=0.5,linestyle=':',zorder=11)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5,zorder=10)
cba = plt.colorbar(ix2,  extend='neither', orientation='vertical',pad=0,shrink=1.0, ax=ax)
cba.set_ticks(np.arange(0,181,30),minor=False)
cba.ax.tick_params(labelsize=8,direction='in',left='False',width=0.2,pad=0.3)
cba.set_label('mm',fontsize=6)
plt.xlabel('Three streamflow gauges')
plt.title('CRU data in MRB')
plt.show()  

##### I will plot the streamflow stations onto the areas
#Plots the data and graphs again
fig = plt.figure(figsize=(10, 3),dpi=250)
ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))  
plot_lev=np.arange(00,181,15)
plot_var=(pr_clim)
ix1=ax.contourf(plot_var.lon,plot_var.lat,plot_var,levels=plot_lev,extend='both',cmap='jet_r',transform=ccrs.PlateCarree(),zorder=1,alpha=0.8)
ix2=ax.contourf(plot_var.lon,plot_var.lat,plot_var,levels=plot_lev,extend='neither',cmap='jet_r',transform=ccrs.PlateCarree(),zorder=1,alpha=0.8)
#This scatterplot will plot the streamflow stations     
plt1=ax.scatter(data['Longitude'],data['Latitude'],color='gold',edgecolor='blue',s=20.0,linewidth=0.5,transform=ccrs.PlateCarree(),zorder=2,alpha=1)
feat1=cfeature.ShapelyFeature(mrb['geometry'],crs=ccrs.PlateCarree(),facecolor='None',edgecolor='k',linewidth=0.7,zorder=13)
#feature 1 is the MRB again
ax.add_feature(feat1)
ax.set_extent([220 ,310,20,55], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linewidth=0.5,zorder=10)
ax.add_feature(cfeature.STATES, linewidth=0.2,alpha=0.5,linestyle=':',zorder=11)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5,zorder=10)
cba = plt.colorbar(ix2,  extend='neither', orientation='vertical',pad=0,shrink=1.0, ax=ax)
cba.set_ticks(np.arange(0,181,30),minor=False)
cba.ax.tick_params(labelsize=8,direction='in',left='False',width=0.2,pad=0.3)
cba.set_label('mm',fontsize=6)
plt.xlabel('Three streamflow gauges')
plt.title('Three Gauges in the MRB')
plt.show()  
    
######## plotted all of the precip and all the streamflow 



### I will now plot a coorelation between PDSI and streamflow 

#Once again need to convert the stream id to lat lon
#This will take alot of code
station_lat_lon=pd.read_csv('lat_lon_points',sep='\t',skiprows=26).iloc[1:,:]
#Run in a station
in_station1=pd.read_csv('sf_Arkansas_River_AT_WICHITA_KS',sep='\t',skiprows=35).iloc[1:,:]
#Changing the station to lat lon
st=in_station1['site_no'].iloc[0]
st_idx=station_lat_lon.site_no.index[station_lat_lon.site_no==st].tolist()[0]
st_lat=float(station_lat_lon.iloc[st_idx,3])
st_lon=float(station_lat_lon.iloc[st_idx,4])
###order the pandas dataframe
in_station1=in_station1.rename(columns={'year_nu':'Year','month_nu':'Month'})
in_station1['date']=pd.to_datetime(in_station1[['Year', 'Month']].assign(DAY=1))
in_station1_wk=in_station1.loc[:,['mean_va','date']]
in_station1_wk.index=in_station1_wk.date
in_station1_wk=in_station1_wk.drop('date',axis=1)
in_station1_ds=in_station1_wk.to_xarray()['mean_va'].astype('float')
in_station1_ds_norm=(in_station1_ds-in_station1_ds.mean('date'))/(in_station1_ds.std('date'))


in_dir='/Users/bam/Documents/fimallanb/stream'
allfiles = glob.glob(in_dir + '/*')#'sf_Mississippi_River_at_Keokuk_IA, sf_Ohio_River_at_Sewickley_PA, sf_Arkansas_River_AT_WICHITA_KS')
namefile=[allfiles[i].split('/')[-1] for i in range(len(allfiles))]
########
in_station1_ds=[]
st_lat=[]
st_lon=[]
for irun in range(len(allfiles)):
    #in_station1=pd.read_csv('/data/Databank/Paleo_data/MRB_Stream_data/main_rivers/sf_Arkansas_River_at_Ralston_OK',sep='\t',skiprows=35).iloc[1:,:]
    in_station1=pd.read_csv(allfiles[irun],sep='\t',skiprows=35).iloc[1:,:]
    #print(in_station1)
    st=in_station1['site_no'].iloc[0]
    st_idx=station_lat_lon.site_no.index[station_lat_lon.site_no==st].tolist()[0]
    st_lat.append(float(station_lat_lon.iloc[st_idx,3]))
    st_lon.append(float(station_lat_lon.iloc[st_idx,4]))
    in_station1=in_station1.rename(columns={'year_nu':'Year','month_nu':'Month'})
    in_station1['time']=pd.to_datetime(in_station1[['Year', 'Month']].assign(DAY=1))
    in_station1_wk=in_station1.loc[:,['mean_va','time']]
    in_station1_wk.index=in_station1_wk.time
    in_station1_wk=in_station1_wk.drop('time',axis=1)
    in_station1_wk=in_station1_wk.asfreq('1MS')
    in_station1_ds.append(in_station1_wk.to_xarray()['mean_va'].astype('float'))

#start the time series 

in_station1_ds=xr.concat(in_station1_ds,'station')
in_station1_ds['station']=namefile
st_lat_ds=xr.DataArray(st_lat,coords=[namefile],dims=['station']).rename('lat')
st_lon_ds=xr.DataArray(st_lon,coords=[namefile],dims=['station']).rename('lon')
in_station1_ds=xr.merge([in_station1_ds,st_lat_ds,st_lon_ds])

NADS=xr.open_dataset('nada_hd2_cl.nc')['pdsi'].sel(time=slice('1900','2005'))
NADS['time']=pd.date_range('1900-01-01',freq='1YS',periods=105)
NADS_wk = NADS.sel(lon=st_lon,lat=st_lat,method='nearest')
ann_wk1 = in_station1_ds.resample(time='YS').mean('time',skipna=False)
sum_wk1 = in_station1_ds.sel(time=in_station1_ds.time.dt.month.isin([6,7,8])).resample(time='1YS').mean('time',skipna=False)
# Re-initialize variables for the updated dataset
# data as 1d numpy array
def check_location_use(data):
    start = None
    current_streak = 0
    indices_updated = []
    for i, value in enumerate(data):
        # Check if the value is not NaN
        if not np.isnan(value):
            if start is None:
                start = i
            current_streak += 1
        else:
            if current_streak >= 10:
                indices_updated.append((start, start + current_streak - 1))
            start = None
            current_streak = 0
    if current_streak >= 10:
        indices_updated.append((start, start + current_streak - 1))
    return(indices_updated)

#Make the Empty list
Ann_ST_cor=[]
Sum_ST_cor=[]
pval=[]
sum_pval=[]
wk_lat=[]
wk_lon=[]
#Make a loop over the dataset
#(Ann_ST_cor, Sum_ST_cor, pval, sum_pval, wk_lat, wk_lon) are initialized. These will store results for annual and summer correlations, p-values from statistical tests, and geographical coordinates (latitude and longitude) of the stations
for ista,sta in  enumerate(in_station1_ds.station.values):
    ann_ind_wk=ann_wk1.isel(station=ista).sel(time=slice('1900','2004'))
    sum_ind_wk=sum_wk1.isel(station=ista).sel(time=slice('1900','2004'))
    workloc=check_location_use(ann_ind_wk['mean_va'].values)
    ann_ind_wk1= xr.merge([ann_ind_wk.isel(time=slice(workloc[i][0],workloc[i][1]+1)) for i in range(len(workloc))])
    sum_ind_wk1= xr.merge([sum_ind_wk.isel(time=slice(workloc[i][0],workloc[i][1]+1)) for i in range(len(workloc))])
    NADS_ind_wk=NADS_wk.isel(lon=ista,lat=ista)
    NADS_ind_wk1=NADS_ind_wk.sel(time=ann_ind_wk1.time)
    #Correlations and linear regressions between NADS_ind_wk1 and both ann_ind_wk1['mean_va'] and sum_ind_wk1['mean_va'] are computed using stats.pearsonr and stats.linregress functions. 
    # functions compute Pearson correlation coefficients and parameters of linear regression, and  p-values.
    cor,cor_p=stats.pearsonr(NADS_ind_wk1,ann_ind_wk1['mean_va'])
    res,_,_,res_p,_=stats.linregress(NADS_ind_wk1,ann_ind_wk1['mean_va'])
    scor,scor_p=stats.pearsonr(NADS_ind_wk1,sum_ind_wk1['mean_va'])
    sres,_,_,sres_p,_=stats.linregress(NADS_ind_wk1,sum_ind_wk1['mean_va'])
    Ann_ST_cor.append(cor)
    Sum_ST_cor.append(scor)
    pval.append(cor_p)
    sum_pval.append(scor_p)
    #coordinates of each station are extracted from ann_wk1 and appended to the wk_lat and wk_lon lists.
    wk_lat.append(ann_wk1.lat[0,ista].values)
    wk_lon.append(ann_wk1.lon[0,ista].values)
    
    #a DataFrame is created from a dictionary where each key-value pair corresponds to a column in the DataFrame. 
    
data_sub_corr_pd=pd.DataFrame(
    {'lat': wk_lat,
     'lon': wk_lon,
     'Ann_ST_cor': Ann_ST_cor,
     'Sum_ST_cor': Sum_ST_cor,
     'pval': pval,
     'sum_pval': sum_pval,
    })

#Plot the figues
fig = plt.figure(figsize=[20,40],dpi=250)
fig.subplots_adjust(hspace=0.4, wspace=0.2)



for ista,sta in  enumerate(in_station1_ds.station.values):
    ann_ind_wk=ann_wk1.isel(station=ista).sel(time=slice('1900','2004'))
    sum_ind_wk=sum_wk1.isel(station=ista).sel(time=slice('1900','2004'))
    
    workloc=check_location_use(ann_ind_wk['mean_va'].values)

    ann_ind_wk1= xr.merge([ann_ind_wk.isel(time=slice(workloc[i][0],workloc[i][1]+1)) for i in range(len(workloc))])
    sum_ind_wk1= xr.merge([sum_ind_wk.isel(time=slice(workloc[i][0],workloc[i][1]+1)) for i in range(len(workloc))])

    NADS_ind_wk=NADS_wk.isel(lon=ista,lat=ista)
    NADS_ind_wk1=NADS_ind_wk.sel(time=ann_ind_wk1.time)


    cor,cor_p=stats.pearsonr(NADS_ind_wk1,ann_ind_wk1['mean_va'])
    res,_,_,res_p,_=stats.linregress(NADS_ind_wk1,ann_ind_wk1['mean_va'])
    
    scor,scor_p=stats.pearsonr(NADS_ind_wk1,sum_ind_wk1['mean_va'])
    sres,_,_,sres_p,_=stats.linregress(NADS_ind_wk1,sum_ind_wk1['mean_va'])


#### plot

    
    ax1 = fig.add_subplot(20, 2, ista+1)

    var1=((NADS_ind_wk1-NADS_ind_wk1.mean('time'))/NADS_ind_wk1.std('time'))
    var2=((ann_ind_wk1['mean_va']-ann_ind_wk1['mean_va'].mean('time'))/ann_ind_wk1['mean_va'].std('time'))
    var3=((sum_ind_wk1['mean_va']-sum_ind_wk1['mean_va'].mean('time'))/sum_ind_wk1['mean_va'].std('time'))
    
    xlab=pd.date_range('1900-01-01',freq='1YS',periods=105)

    ax1.plot(NADS_ind_wk1.time,var1,color='orange',label='PDSI',linewidth=1)
    ax1.plot(NADS_ind_wk1.time,var2,color='blue',label=f'Ann_ST cor={cor:.2f} pval={cor_p:.3f}',linewidth=1)
    ax1.plot(NADS_ind_wk1.time,var3,color='red',label=f'Sum_ST cor={scor:.2f} pval={scor_p:.3f}',linewidth=1)
    ax1.grid(axis='x',which='major',color='grey',linestyle='--', alpha=0.5,linewidth=0.5)

####plot parameters
    ax1.set_xlim([xlab[0],xlab[-1]])
    
    ax1.set_ylim([-3.5,3.5])
    ax1.set_yticks(np.arange(-3,3.1,1))
    ax1.set_yticklabels(['','-2','','0','','2',''])
    
    ax1.axhline(y = 0, color = 'grey', linestyle = 'dashed',alpha=0.5,linewidth=0.5)
    ax1.tick_params(axis='y',which='major',direction = 'in',length =3 ,labelsize=6,left=True,right=True)
    ax1.tick_params(axis='x',which='major',direction = 'in',length =3 ,labelsize=6,top=True,bottom=True)

    if(ista==26):
        ax1.legend(fontsize=6,loc='upper right')
    else:
        ax1.legend(fontsize=6,loc='upper left')

    st_lat=ann_ind_wk.lat[0]
    st_lon=ann_ind_wk.lon[0]
    ax1.set_title(f'{sta[3:]}  Lat:{st_lat:.2f} Lon:{st_lon:.2f}',  fontsize=8, loc='left', pad=6)

    
###### STREAMFLOW COOR PLOTTED  ^^^^^^^^

### Plot a streamflow time series
# Read the Arkansas river in wichita kansas,
df_adjusted = pd.read_csv('sf_Arkansas_River_AT_WICHITA_KS', sep='\t', skiprows=35, header=None)
#assignin column names
df_adjusted.columns = ['agency_cd', 'site_no', 'parameter_cd', 'ts_id', 'year', 'month', 'mean_discharge']
#convert 'year' and 'month' to integers, and 'mean_discharge' to float
df_adjusted['year'] = pd.to_numeric(df_adjusted['year'], errors='coerce')
df_adjusted['month'] = pd.to_numeric(df_adjusted['month'], errors='coerce')
df_adjusted['mean_discharge'] = pd.to_numeric(df_adjusted['mean_discharge'], errors='coerce')
#remove any rows with NaN values after conversion 
df_adjusted.dropna(subset=['year', 'month', 'mean_discharge'], inplace=True)
# Convert year and month columns to a single datetime column
df_adjusted['date'] = pd.to_datetime(df_adjusted[['year', 'month']].assign(day=1))
# Set the datetime as the index of the DataFrame
df_adjusted.set_index('date', inplace=True)
# Plotting the time series of the streamflow discharge
# Assuming 'df_adjusted' is your DataFrame after initially loading and setting the index
filtered_df = df_adjusted[(df_adjusted.index.year >= 1940) & (df_adjusted.index.year <= 2000)]
#This is the plots and labels
plt.figure(figsize=(12, 5))
plt.plot(filtered_df.index, filtered_df['mean_discharge'], label='Mean Streamflow Discharge (1940-2000)', color='purple')
plt.title('Streamflow Discharge at Arkansas River at Wichita, KS (1940-2000)')
plt.xlabel('Date')
plt.ylabel('Mean Streamflow Discharge (cfs)')
plt.grid(True)
plt.legend()
plt.show()

##

#Same code but for Keokuk IA
df_adjusted = pd.read_csv('sf_Mississippi_River_at_Keokuk_IA', sep='\t', skiprows=35, header=None)
df_adjusted.columns = ['agency_cd', 'site_no', 'parameter_cd', 'ts_id', 'year', 'month', 'mean_discharge']
df_adjusted['year'] = pd.to_numeric(df_adjusted['year'], errors='coerce')
df_adjusted['month'] = pd.to_numeric(df_adjusted['month'], errors='coerce')
df_adjusted['mean_discharge'] = pd.to_numeric(df_adjusted['mean_discharge'], errors='coerce')
df_adjusted.dropna(subset=['year', 'month', 'mean_discharge'], inplace=True)
df_adjusted['date'] = pd.to_datetime(df_adjusted[['year', 'month']].assign(day=1))
df_adjusted.set_index('date', inplace=True)
filtered_df = df_adjusted[(df_adjusted.index.year >= 1940) & (df_adjusted.index.year <= 2000)]
plt.figure(figsize=(12, 5))
plt.plot(filtered_df.index, filtered_df['mean_discharge'], label='Mean Streamflow Discharge (1940-2000)', color='purple')
plt.title('Streamflow Discharge at Mississippi RIver at Keokuk IA (1940-2000)')
plt.xlabel('Date')
plt.ylabel('Mean Streamflow Discharge (cfs)')
plt.grid(True)
plt.legend()
plt.show()

######

#Same code but for Ohio River in Sewickley PA
df_adjusted = pd.read_csv('sf_Ohio_River_at_Sewickley_PA', sep='\t', skiprows=35, header=None)
df_adjusted.columns = ['agency_cd', 'site_no', 'parameter_cd', 'ts_id', 'year', 'month', 'mean_discharge']
df_adjusted['year'] = pd.to_numeric(df_adjusted['year'], errors='coerce')
df_adjusted['month'] = pd.to_numeric(df_adjusted['month'], errors='coerce')
df_adjusted['mean_discharge'] = pd.to_numeric(df_adjusted['mean_discharge'], errors='coerce')
df_adjusted.dropna(subset=['year', 'month', 'mean_discharge'], inplace=True)
df_adjusted['date'] = pd.to_datetime(df_adjusted[['year', 'month']].assign(day=1))
df_adjusted.set_index('date', inplace=True)
filtered_df = df_adjusted[(df_adjusted.index.year >= 1940) & (df_adjusted.index.year <= 2000)]
plt.figure(figsize=(12, 5))
plt.plot(filtered_df.index, filtered_df['mean_discharge'], label='Mean Streamflow Discharge (1940-2000)', color='purple')
plt.title('Streamflow Discharge at Ohio River at Sewickley PA (1940-2000)')
plt.xlabel('Date')
plt.ylabel('Mean Streamflow Discharge (cfs)')
plt.grid(True)
plt.legend()
plt.show()




# Load precipitation data with xarray
pr = xr.open_dataset('cru_ts4.07.1901.2022.pre.dat.nc').pre

# Restricting data to the period from 1940 to 2000
pr_time_series = pr.sel(time=slice('1940-01-01', '2000-12-31')).mean(dim=['lat', 'lon'])

# Plotting the time series
fig, ax = plt.subplots(figsize=(10, 5))
pr_time_series.plot(ax=ax, label='Monthly Average Precipitation')

# Calculate and plot the trend line
# Convert time index to a numerical format for regression
time_numeric = np.arange(len(pr_time_series))
slope, intercept = np.polyfit(time_numeric, pr_time_series, 1)
trend_line = slope * time_numeric + intercept
ax.plot(pr_time_series.time, trend_line, label='Trend Line', color='red')

# Customize the plot
ax.set_title('Time Series of Average Precipitation (1940-2000)')
ax.set_xlabel('Time')
ax.set_ylabel('Precipitation (mm)')
ax.legend()
plt.grid(True)
plt.show()


