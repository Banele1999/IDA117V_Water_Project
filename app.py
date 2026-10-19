import json
from pathlib import Path
import pandas as pd,streamlit as st,joblib
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Water Reliability Screening',page_icon='💧')
st.title('Household Water Reliability Screening')
st.info('Research demonstration only. Not suitable for denying services, ranking individuals, or automated resource allocation. Survey data reflect 2023 and predictions are associations, not forecasts.')
meta_path=ROOT/'models'/'metadata.json'
if not meta_path.exists():st.error('Run python train.py first to generate a model.');st.stop()
meta=json.loads(meta_path.read_text());st.caption('Champion: '+meta['champion'])
@st.cache_data
def options():
 d=pd.read_csv(ROOT/'data'/'ghs-2023-hhold-v1.csv',usecols=meta['features'],low_memory=False)
 return {c:sorted(d[c].dropna().astype(str).unique().tolist()) for c in meta['features']}
opts=options();values={}
with st.form('predict'):
 for c in meta['features']:values[c]=st.selectbox(c,opts[c],help='Choose a category observed in the GHS 2023 dataset')
 submitted=st.form_submit_button('Estimate interruption likelihood')
if submitted:
 x=pd.DataFrame([values]);
 if meta['type']=='sklearn':p=float(joblib.load(ROOT/'models'/'pipeline.joblib').predict_proba(x)[0,1])
 else:
  import torch
  from neural import NeuralNet
  pp=joblib.load(ROOT/'models'/'preprocessor.joblib');ck=torch.load(ROOT/'models'/'neural.pt',map_location='cpu',weights_only=True);net=NeuralNet(ck['input_dim'],ck['residual']);net.load_state_dict(ck['state_dict']);net.eval()
  with torch.no_grad():p=float(torch.sigmoid(net(torch.tensor(pp.transform(x).astype('float32')))).item())
 st.metric('Estimated probability of reported interruption',f'{p:.1%}');st.write('Predicted class: **'+('Interruption reported' if p>=.5 else 'No interruption reported')+'**')
 st.caption('Probability is model-estimated and may be miscalibrated. Review service records and community evidence before decisions.')
