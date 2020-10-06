import tensorflow as tf 
import numpy as np

def ocs_wts(weights):
    split_threshold = 0.5
    num_channels = weights.shape[0]
    print('num_channel', num_channels)
    ocs_channels = int(np.ceil(0.1 * num_channels))
    print('ocs_channels', ocs_channels)
    
    # Which act channels to copy
    in_channels_to_copy = []
    # Mapping from newly added channels to the orig channels they split from
    orig_idx_dict = {}

    for c in range(ocs_channels):
        # pick the channels with the largest max values
        # axes = list(range(weights.ndim))
        # axes.remove(axis)
        # find max along rows
        max_per_channel = np.max(np.abs(weights), axis=1)
        # Sort and compute which channel to split
        idxs = np.flip(np.argsort(max_per_channel), axis=0)
        split_idx = idxs[0]

        # Split channel
        # ch_slice = weights[:, split_idx:(split_idx+1), :, :].copy()
        ch_slice = weights[split_idx:(split_idx+1), :].copy()

        ch_slice_half = ch_slice / 2.
        ch_slice_zero = np.zeros_like(ch_slice)
        split_value = np.max(ch_slice) * split_threshold

        ch_slice_1 = np.where(np.abs(ch_slice) > split_value, ch_slice_half, ch_slice)
        ch_slice_2 = np.where(np.abs(ch_slice) > split_value, ch_slice_half, ch_slice_zero)

        # if not grid_aware:
        #     ch_slice_1 = np.where(np.abs(ch_slice) > split_value, ch_slice_half, ch_slice)
        #     ch_slice_2 = np.where(np.abs(ch_slice) > split_value, ch_slice_half, ch_slice_zero)
        # else:
        #     ch_slice_half *= w_scale
        #     ch_slice_1 = np.where(np.abs(ch_slice) > split_value, ch_slice_half-0.25, ch_slice*w_scale) / w_scale
        #     ch_slice_2 = np.where(np.abs(ch_slice) > split_value, ch_slice_half+0.25, ch_slice_zero)    / w_scale

        # weights[:, split_idx:(split_idx+1), :, :] = ch_slice_1
        # weights = np.concatenate((weights, ch_slice_2), axis=axis)
        weights[split_idx:(split_idx+1), :] = ch_slice_1
        weights = np.concatenate((weights, ch_slice_2), axis=0)

        # Record which channel was split
        if split_idx < num_channels:
            in_channels_to_copy.append(split_idx)
            orig_idx_dict[num_channels+c] = split_idx
        else:
            idx_to_copy = orig_idx_dict[split_idx]
            in_channels_to_copy.append(idx_to_copy)
            orig_idx_dict[num_channels+c] = idx_to_copy

        # in_channels_to_copy.append(split_idx)

    return weights, in_channels_to_copy