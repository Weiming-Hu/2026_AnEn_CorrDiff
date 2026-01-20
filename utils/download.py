import os
import io
import gzip
import zipfile
import requests

from .roots import DATA_ROOT

# The extent extracted for western US from PRISM
west_us_extent = {
    'xmin': -125,
    'xmax': -116,
    'ymin': 32,
    'ymax': 41.5,
}

def touch_prism(dt, variable):
    assert variable in ['ppt', 'tmean'], 'Currently can download ppt or tmean'
    
    root_folder = os.path.join(DATA_ROOT, 'input/PRISM/daily_stable')
    
    url_template = \
        'https://data.prism.oregonstate.edu/daily/' + variable + \
        '/{}/PRISM_' + variable + '_stable_4kmD2_{}_bil.zip'
    
    url = url_template.format(
        dt.strftime(format='%Y'),
        dt.strftime(format='%Y%m%d')
    )
    
    base_name = url.split('/')[-1].rstrip('.zip')
    output_folder = os.path.join(root_folder, base_name)
    
    if not os.path.exists(output_folder):
        response = requests.get(url)

        if response.status_code == 200:
            os.makedirs(output_folder, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                zip_ref.extractall(path=output_folder)
        else:
            print(f'Errored ({url}), reasons: {response.reason}')

    return os.path.join(output_folder, base_name + '.bil')

    