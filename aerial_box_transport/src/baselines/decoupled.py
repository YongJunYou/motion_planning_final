"""Optional baseline: plan the base trajectory first, then arm and squeeze on top.

Shows the value of joint whole-body planning. Optional, implemented in Step 3
only if time allows (spec section 8). See ROADMAP.md.
"""


def plan_decoupled(*args, **kwargs):
    raise NotImplementedError("Decoupled baseline is optional, Step 3 (see ROADMAP.md).")
