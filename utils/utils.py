import struct as st

def save_data(tensor, path, is_act=False, to_int=False, to_hex=False, output_dir=None, q=0.0):
    # if path.split('.')[-1] not in ['1']:
    #     print(f'Skipping {path}')
    #     return
    assert is_act + to_int + to_hex <= 1, path

    def convert_hex(x):
        return '%X' % st.unpack('H', st.pack('e', x))

    path = f'{output_dir}/{path}'
    print(f'Saving {path}')

    if to_hex:
        with open(f'{path}.txt', 'w') as f:
            print('\n'.join(
                f'{convert_hex(num.item())}'
                for num in tensor.view(-1)
            ), file=f)
    else:
        if to_int:
            tensor = tensor.round().int()
        elif is_act:
            tensor = tensor.mul(2**q).round().int()
        with open(f'{path}.txt', 'w') as f:
            print('\n'.join(
                f'{num.item()}'
                for num in tensor.view(-1)
            ), file=f)