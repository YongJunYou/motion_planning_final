"""Read the teammate's scene assets (box, desk, rack) to adapt OCP parameters.

Reports each asset's world bounding box (size + where its surfaces sit) and any
authored physics mass/density, so the OCP's box size / mass / pick-place heights
can be matched to the scene WITHOUT editing the teammate's files.

Run: conda run -n am_dualarm python src/model/parse_scene.py
"""
from pxr import Usd, UsdGeom, UsdPhysics, Gf

REPO = "/home/jaewoo/Research/motion_planning_final"
ASSETS = {
    "box  (cubebox_a01)": f"{REPO}/box/cubebox_a01/cubebox_a01.usd",
    "desk (desk_01)": f"{REPO}/surroundings/desk_01/desk_01_inst_base.usd",
    "rack (rack_l01)": f"{REPO}/surroundings/rack_l01/rack_l01_inst_base.usd",
}


def report(name, path):
    stage = Usd.Stage.Open(path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                           [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = bb.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    size = [(hi[i] - lo[i]) for i in range(3)]
    cen = [0.5 * (hi[i] + lo[i]) for i in range(3)]
    print(f"\n=== {name} ===  metersPerUnit={mpu}")
    print(f"  bbox min  {[round(lo[i], 4) for i in range(3)]}")
    print(f"  bbox max  {[round(hi[i], 4) for i in range(3)]}")
    print(f"  size (m)  {[round(size[i] * mpu, 4) for i in range(3)]}   (raw {[round(s,3) for s in size]})")
    print(f"  center    {[round(cen[i], 4) for i in range(3)]}")
    # physics mass / density on any prim
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(p)
            mass = m.GetMassAttr().Get()
            dens = m.GetDensityAttr().Get()
            if mass or dens:
                print(f"  MassAPI on {p.GetName()}: mass={mass} density={dens}")
        if p.HasAPI(UsdPhysics.CollisionAPI):
            pass


for name, path in ASSETS.items():
    try:
        report(name, path)
    except Exception as e:
        print(f"\n=== {name} === FAILED: {e}")
