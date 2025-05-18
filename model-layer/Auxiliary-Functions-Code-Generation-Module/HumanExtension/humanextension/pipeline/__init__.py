from humanextension.pipeline.single.base.direct import (
    implement_direct_humaneval,
    implement_direct_humanextension,
)
from humanextension.pipeline.single.instruct.plain import (
    implement_instruct_humaneval,
    implement_instruct_humanextension,
)
from humanextension.pipeline.single.instruct.step_by_step import (
    implement_instruct_step_by_step_humanextension, )
from humanextension.pipeline.single.base.irrelevant import (
    implement_irrelevant_humanextension, )
from humanextension.pipeline.single.base.oracle import implement_oracle_humanextension
from humanextension.pipeline.single.base.step_by_step import (
    implement_step_by_step_humanextension, )

from humanextension.pipeline.multiple.multiple_auxiliary_functions import (
    implement_multiple_auxiliary_functions_humanextension, )
