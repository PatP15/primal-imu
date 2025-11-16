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

# Track statistics
stats = {
    'total': len(seqfiles),
    'already_has_jts': 0,
    'missing_keys': 0,
    'load_errors': 0,
    'process_errors': 0,
    'successful': 0
}

for seqfile in tqdm(seqfiles):
    
    ## load data
    try:
        data = np.load(seqfile, allow_pickle=True)
        
        # Skip if jts_body already exists
        if 'jts_body' in data:
            stats['already_has_jts'] += 1
            continue
        
        # Check for all required keys before processing
        required_keys = ['trans', 'root_orient', 'pose_body', 'betas', 'mocap_frame_rate']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"Skipping {seqfile}: missing keys {missing_keys}")
            stats['missing_keys'] += 1
            continue
            
        trans = torch.tensor(data['trans']).float().cuda() #[t,3]
    except Exception as e:
        print(f"Error loading {seqfile}: {e}")
        stats['load_errors'] += 1
        continue
    
    try:
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
        
        # Explicitly delete GPU tensors to free memory
        del trans, glorot_aa, pose_body, betas, transf_transl, transf_rotmat
        del xb, xb_new, jts_body
        torch.cuda.empty_cache()
        
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
        stats['successful'] += 1
        
    except Exception as e:
        print(f"Error processing {seqfile}: {e}")
        # Clean up GPU memory even on error
        torch.cuda.empty_cache()
        stats['process_errors'] += 1
        continue

# Print summary
print("\n" + "="*60)
print("CONVERSION SUMMARY")
print("="*60)
print(f"Total files found:           {stats['total']}")
print(f"Already had jts_body:        {stats['already_has_jts']}")
print(f"Successfully processed:       {stats['successful']}")
print(f"Skipped (missing keys):       {stats['missing_keys']}")
print(f"Load errors:                 {stats['load_errors']}")
print(f"Processing errors:           {stats['process_errors']}")
print(f"\nValid files (with jts_body): {stats['already_has_jts'] + stats['successful']}")
print(f"Invalid files (skipped):      {stats['missing_keys'] + stats['load_errors'] + stats['process_errors']}")
print("="*60)



        