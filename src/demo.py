import h5py

with h5py.File('_hdf5/label/adpit/dev/official.h5', 'r') as f:
    print("文件结构:")
    def print_attrs(name, obj):
        print(name)
    f.visititems(print_attrs)