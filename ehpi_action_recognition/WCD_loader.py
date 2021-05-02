import pandas as pd

def read_csv_annots(file):
  annots = pd.read_csv(file)
  action_vids_as_frames = []
  for action_vid_name in annots['cut_name']:
    vid_infos = action_vid_name.split('_')
    vid_name = vid_infos[0]
    first_frame = int (vid_infos[1])
    last_frame = int (vid_infos[2])
    action_frame_paths = []
    for num in range(first_frame, last_frame+1):
      action_frame_paths.append(vid_name + '-' + f'{num:03d}' + '.jpg')
    action_vids_as_frames.append(action_frame_paths)
  return action_frame_paths
