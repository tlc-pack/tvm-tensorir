import sys
from typing import Dict, List, Tuple

import pytest
import tvm
from tvm import meta_schedule as ms
from tvm.ir import IRModule
from tvm.meta_schedule.integration import ExtractedTask
from tvm.meta_schedule.testing import get_network
from tvm.runtime import NDArray


@pytest.mark.parametrize("num_layers", [18, 34, 50, 101, 152])
def test_meta_schedule_extract_from_resnet(num_layers: int):
    mod, params, input_shape, output_shape = get_network(
        name="resnet-" + str(num_layers),
        batch_size=1,
        layout="NHWC",
        dtype="float32",
    )
    extracted_tasks = ms.integration.extract_task(mod, params, target="llvm")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
