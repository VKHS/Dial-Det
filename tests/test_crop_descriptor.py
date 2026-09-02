import numpy as np

from dial_det.adapters.crop_descriptor import DESCRIPTOR_ID, dial48_crop_descriptor


def test_crop_descriptor_is_deterministic_48d():
    image=np.arange(16*16*3,dtype=np.uint8).reshape(16,16,3)
    first=dial48_crop_descriptor(image)
    second=dial48_crop_descriptor(image.copy())
    assert DESCRIPTOR_ID=='dial48-reference-v1'
    assert first.shape==(48,)
    assert np.allclose(first,second)
    assert np.isfinite(first).all()
