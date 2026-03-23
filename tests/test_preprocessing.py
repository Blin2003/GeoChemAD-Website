import numpy as np
import pandas as pd

from geochemad.preprocessing import PreprocessingConfig, prepare_samples


def test_prepare_samples_handles_abnormal_values_and_ilr():
    frame = pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0],
            "Y": [4.0, 5.0, 6.0],
            "SAMPLEID": ["a", "b", "c"],
            "Au_ppm": [0.1, -9999.0, 0.3],
            "Cu_ppm": [1.0, 2.0, 0.0],
            "Ni_ppm": [3.0, 4.0, 5.0],
        }
    )
    prepared = prepare_samples(frame, "Au", PreprocessingConfig(compositional_transform="ilr"))
    assert prepared.coords.shape == (3, 2)
    assert prepared.features.shape[0] == 3
    assert np.isfinite(prepared.features).all()
