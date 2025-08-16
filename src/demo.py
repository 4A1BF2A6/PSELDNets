# import h5py

# with h5py.File('_hdf5/label/adpit/dev/official.h5', 'r') as f:
#     print("文件结构:")
#     def print_attrs(name, obj):
#         print(name)
#     f.visititems(print_attrs)

import os

csv_path = '_hdf5/data/24000fs/wav/dev/STARSS22_10sChunklen_10sHoplen_train.csv'  # 你的csv文件路径

with open(csv_path, 'r') as f:
    for i, line in enumerate(f, 1):
        audio_path = line.strip().split(',')[0]
        if not os.path.exists(audio_path):
            print(f'第{i}行文件不存在: {audio_path}')

import soundfile as sf
import glob

for f in glob.glob('datasets/STARSS22/foa_dev/*.wav'):
    try:
        data, sr = sf.read(f)
    except Exception as e:
        print(f"Error reading: {f}，错误信息: {e}")