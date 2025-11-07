import os
import geopandas as gpd
import pandas as pd
from VotE import config; config.vote_db()
from VotE.streamflow import export_streamflow as es
from VotE import export_ops as eo


#------------ Download streamflow data from VotE

# # Relevant gages for meeting Susan's informed objectives
# 15096645837 - first downstream of dam
# 14986503931 - first downstream of animus confluence
# 14991549932 - last animus gage, just upstream of sj confluence
# 15059757555 - next upstream animus (2nd)
# 15058713264 - next upstream animus, at Durango, before confluence with Lightner Creek
# 14818767598 - four corners (downstream of Farmington)
# 14643897418 - san juan downstream of four corners (use for flood control   )

sj_relevant = [15096645837, 14986503931, 14991549932, 15059757555, 15058713264, 14818767598, 14643897418]

# Set path to store basin tokens
out = r'X:\Research\DeepReservoir\finalize_model\gages'
path_ts = os.path.join(out, 'timeseries')

sdf = es.get_streamflow_timeseries(sj_relevant)

renamer = {
    15096645837: 'SJ @ Archuleta',
    14986503931: 'SJ @ Farmington',
    14991549932: 'Animus @ Farmington',
    15059757555: 'Animus_us1',
    15058713264: 'Animus @ Durango',
    14818767598: 'SJ @ Shiprock',
    14643897418: 'SJ @ Bluff'
}


CMS_TO_CFS = 35.3146667  # 1 m³/s = 35.3146667 ft³/s

# start from your df
df2 = (
    sdf.drop(columns=['q_quality'], errors='ignore')  # drop quality col if present
      .assign(
          date=pd.to_datetime(sdf['date']),
          name=sdf['id_gage'].astype('int64').map(renamer).fillna(sdf['id_gage'].astype(str)),
          q_cfs=pd.to_numeric(sdf['q_cms'], errors='coerce') * CMS_TO_CFS
      )
)

# wide format: index = date, columns = gage names, values = cfs
# if there can be duplicate (date, gage) rows, switch aggfunc to "mean" (or what you prefer)
wide = (
    df2.pivot_table(index='date', columns='name', values='q_cfs', aggfunc='first')
       .sort_index()
)
wide.columns.name = None  # drop the columns index name

wide.to_csv(r'X:\Research\DeepReservoir\finalize_model\gages\all_gages.csv')



#------------ Download rivernetwork

rn =  eo.get_rivnetwork(id_reach=14477553321, id_outlet=None, query_cols=["id_reach", "geom", "length_km", "dist_to_outlet_km", 'drainarea_ds_km2'], da_thresh=1000)
rn.to_file(r'X:\Research\DeepReservoir\finalize_model\gis\powell_rn.gpkg', driver='GPKG')


#------------ Download gages
gage_params = {'within' : 13612257818}
gages = es.gage_selector(gage_params)
g = es.get_gages(gages)
g.to_file(r'X:\Research\DeepReservoir\finalize_model\gis\all_gages.gpkg', driver='GPKG')

tribs = {
    'Powell Inflow' : 14477553321, # no
    'Chinle Creek' : 14677305357, # one year
    'Montezuma Creek' : 14757867273, # 8 years
    'McElmo Creek' : 14788317330, # 2 years
    'Mancos River' : 14830599623, # 1-2 years
    'Chaco River' : 14900547885, # full
    'La Plata River' : 14981979910, # full
    'Animus River' : 14989113941, # full
    'Canyon Largo' : 15071937941, # 3-4 years, old
    'Navajo Outflow' : 15114741840,
    }

all_tribs = []
for t in tribs:
    print(t)
    this_basin = eo.delineate_basin(id_reach=tribs[t])
    this_basin['name'] = t
    all_tribs.append(this_basin)
all_basins = pd.concat(all_tribs)
all_basins['name'] = [k for k in tribs]
all_basins.to_file(r'X:\Research\DeepReservoir\finalize_model\gis\trib_basins.gpkg', driver='GPKG')