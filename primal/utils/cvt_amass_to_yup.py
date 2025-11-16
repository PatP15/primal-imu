"""AMASS is by default Z-up. 
This script is to transform the body parameters to Y-up.
"""

from primal.utils.mop_repr import SMPLXParser
import os.path as osp
import os, glob
import numpy as np
import torch
from tqdm import tqdm

# load raw amass sequence files - use environment variables or defaults
amass_path = os.getenv('AMASS_DATA_PATH', 'datasets/AMASS/AMASS_SMPLX_NEUTRAL')
seqfiles = sorted(glob.glob(osp.join(amass_path, '*/*/*.npz'), recursive=True))

# load smplx parser - use environment variables or defaults
smplx_path = os.getenv('MODEL_REGISTRY_PATH', 'model-registry')
smplx_parser = SMPLXParser(osp.join(smplx_path,"models/SMPLX/neutral/SMPLX_neutral.npz"),
                                   osp.join(osp.dirname(osp.abspath(__file__)),"SSM2.json"),
                                   num_betas=16)
smplx_parser.eval()
smplx_parser.cuda()


# main loop
print(f"Found {len(seqfiles)} files to process")
for seqfile in tqdm(seqfiles):
    
    ## load data
    try:
        data = np.load(seqfile, allow_pickle=True)
        
        # Skip if jts_body already exists
        if 'jts_body' in data:
            continue
        
        # Check for all required keys before processing
        required_keys = ['trans', 'root_orient', 'pose_body', 'betas', 'mocap_frame_rate']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"Skipping {seqfile}: missing keys {missing_keys}")
            continue
            
        trans = torch.tensor(data['trans']).float().cuda() #[t,3]
    except Exception as e:
        print(f"Error loading {seqfile}: {e}")
        continue
    glorot_aa = torch.tensor(data['root_orient']).float().cuda()#[t,3]
    pose_body = torch.tensor(data['pose_body']).float().cuda() #[t,63]
    nt = trans.shape[0]
    betas = torch.tensor(data['betas']).float().cuda().repeat(nt,1) #[t,3]
    
    ## process data (transform to Y-up)
    ## to change Z-up to Y-up, we just need to rotate x axis by 90 deg
    transf_transl = torch.zeros(1,1,3).cuda() #[0,0,0]
    transf_rotmat = torch.tensor(
        [ [1.0000000,  0.0000000,  0.0000000],
          [0.0000000,  0.0000000, -1.0000000],
          [0.0000000,  1.0000000,  0.0000000 ]]
    ).unsqueeze(0).float().cuda()
    
    ## update smplx params
    xb = torch.cat([trans, glorot_aa,pose_body],dim=-1)
    xb_new = smplx_parser.update_transl_glorot(transf_rotmat, transf_transl, betas,xb)
    jts_body = smplx_parser.forward_smplx(betas, xb_new[:,:3], xb_new[:,3:6], xb_new[:,6:69],returntype='jts')[:,:22]

    ## save data
    trans_new = xb_new[:,:3].detach().cpu().numpy()
    glorot_aa_new = xb_new[:,3:6].detach().cpu().numpy()
    jts_body_np = jts_body.detach().cpu().numpy()
    output_data = {}
    output_data['mocap_frame_rate'] = data['mocap_frame_rate']
    output_data['trans'] = trans_new
    output_data['root_orient'] = glorot_aa_new
    output_data['jts_body'] = jts_body_np
    output_data['betas'] = data['betas']
    output_data['pose_body'] = data['pose_body']
    
    # Copy optional fields if they exist
    for key in ['pose_hand', 'pose_jaw', 'pose_eye']:
        if key in data:
            output_data[key] = data[key]
    
    ## solve output path
    ## save in-place (overwrite existing file with jts_body added)
    outputseqfile = seqfile
    outputseqdir = osp.dirname(outputseqfile)

    os.makedirs(outputseqdir, exist_ok=True)

    np.savez(outputseqfile, **output_data)

print("\nConversion complete! All files now have jts_body key.")



        