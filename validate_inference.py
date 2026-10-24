"""Native TeleopIT parity audit; no production inference changes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


def main():
    p=argparse.ArgumentParser()
    for key in ('source','weights','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.adapter=Path(__file__).resolve().parent
    pins=json.loads((a.adapter/'native-sources.json').read_text())
    for name,expected in pins['files'].items():
        assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==expected,name
    sys.path[:0]=[str(a.adapter.resolve()),str(a.source.resolve()),str((a.adapter/'motion-challenge/src').resolve())]
    from adapter import TeleopIT
    from motion_challenge.sdk import StatePacket, ReferenceFrame, BodyState
    from teleopit.interfaces import RobotState
    from teleopit.controllers.observation import VelCmdObservationBuilder
    from teleopit.controllers.rl_policy import RLPolicyController
    sim = ModuleType("teleopit.sim")
    sim.__path__ = [str((a.source / "teleopit/sim").resolve())]
    sys.modules["teleopit.sim"] = sim  # Avoid unrelated simulator/retargeter startup.
    from teleopit.controllers.reference_processing import compute_anchor_velocities, align_reference_yaw
    rt=json.loads((a.adapter/'runtime.json').read_text());rt['weights']=str(a.weights.resolve())
    assert rt['source_commit']==pins['source_commit']
    policy=TeleopIT(a.adapter.resolve(),rt)
    cfg=yaml.safe_load((a.source/'teleopit/configs/robot/g1.yaml').read_text())
    for field,native in [('default',cfg['default_angles']),('scale',cfg['action_scale']),('kp',cfg['kps']),('kd',cfg['kds'])]:
        np.testing.assert_allclose(getattr(policy,field),np.asarray(native),rtol=1e-7,atol=1e-7)
    defaults=yaml.safe_load((a.source/'teleopit/configs/default.yaml').read_text())
    assert defaults['policy_hz']==50 and rt['control_period_s']==.02
    assert 'reference_velocity_smoothing_alpha' not in defaults and 'reference_anchor_velocity_smoothing_alpha' not in defaults
    xml=a.adapter/'motion-challenge/robots/g1-29dof-cpu-v0.xml'
    builder=VelCmdObservationBuilder(dict(cfg,xml_path=str(xml.resolve()),anchor_body_name='torso_link',default_dof_pos=cfg['default_angles']))
    native=RLPolicyController({'policy_path':str(a.weights.resolve()),'device':'cpu','action_scale':cfg['action_scale'],'default_dof_pos':cfg['default_angles'],'clip_range':[-10.,10.]})
    # Same original CPU ONNX weights/provider; isolate input/state semantics from thread scheduling.
    native._session=policy.session
    tree=ast.parse((a.source/'teleopit/sim/runtime_components.py').read_text())
    fn=next(m for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='PolicyStepRunner' for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='_compute_motion_joint_vel')
    scope={'np':np}
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+[fn],[]),'native_joint_velocity','exec'),scope)
    joint_velocity=scope['_compute_motion_joint_vel']
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    canonical=tuple(model.joint(i).name for i in range(1,model.njnt))
    assert tuple(policy.names)==canonical
    rng=np.random.default_rng(241024)
    shuffle=rng.permutation(29);names=tuple(np.asarray(canonical)[shuffle]);order=[names.index(n) for n in canonical]
    errors={k:0. for k in ['observation','history','same_input_network','action','target','synthetic_target','synthetic_history','fixed_initial_alignment']}
    def compare(k,x,y):errors[k]=max(errors[k],float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))
    def bodies(qpos):
        data.qpos[:]=qpos;mujoco.mj_kinematics(model,data)
        return tuple(BodyState(model.body(i).name,tuple(data.xpos[i]),tuple(data.xquat[i][[1,2,3,0]]),(0.,)*3,(0.,)*3) for i in range(1,model.nbody))
    max_action=0.
    for reset in range(4):
        native.reset();builder.reset();previous=None;last=np.zeros(29,np.float32)
        reference=np.r_[rng.normal(0,.5,2),.8,R.random(random_state=rng).as_quat()[[3,0,1,2]],np.asarray(cfg['default_angles'])+rng.normal(0,.1,29)]
        for step in range(40):
            reference[:3]+=rng.normal(0,.005,3)
            rot=R.from_quat(reference[3:7][[1,2,3,0]])*R.from_rotvec(rng.normal(0,.025,3))
            reference[3:7]=rot.as_quat()[[3,0,1,2]];reference[7:]+=rng.normal(0,.01,29)
            actual=reference.copy();actual[7:]+=rng.normal(0,.1,29)
            ar=rot*R.from_rotvec(rng.normal(0,.2,3));actual[3:7]=ar.as_quat()[[3,0,1,2]]
            if step==0:actual=reference.copy();ar=rot
            qd=rng.normal(0,.8,29);omega=rng.normal(0,.5,3);world_omega=ar.apply(omega)
            state=StatePacket(step*.02,names,tuple(actual[7:][shuffle]),tuple(qd[shuffle]),tuple(actual[:3]),tuple(ar.as_quat()),(0.,)*3,tuple(world_omega),bodies(actual))
            ref=ReferenceFrame(step*.02,step*.02,tuple(reference[7:][shuffle]),tuple(reference[:3]),tuple(rot.as_quat()),(0.,)*29,(0.,)*3,(0.,)*3,bodies(reference))
            refs=(ref,)
            if step==0:
                policy.reset(state,refs,reset)
                aligned,*_=align_reference_yaw(reference,actual[3:7].astype(np.float32),None,None,target_root_xy=actual[:2].astype(np.float32))
                compare('fixed_initial_alignment',aligned,reference)
            nstate=RobotState(actual[7:].astype(np.float32),qd.astype(np.float32),actual[3:7].astype(np.float32),omega.astype(np.float32),step*.02,actual[:3].astype(np.float32))
            lv,av=compute_anchor_velocities(builder._base,reference,previous,29,50.)
            vel=joint_velocity(NS(num_actions=29,last_retarget_qpos=previous,policy_hz=50.),reference)
            expected=builder.build(nstate,reference.astype(np.float32),vel,last,lv,av)
            inputs=policy.observations(state,refs)
            expected_raw=native.compute_action(expected)
            compare('observation',inputs['obs'][0],expected)
            compare('history',inputs['obs_history'][0],native.get_debug_inputs()['obs_history'])
            compare('same_input_network',policy.infer({'obs':expected[None],'obs_history':native.get_debug_inputs()['obs_history'][None]})['actions'][0],expected_raw)
            fn=policy.observations;policy.observations=lambda *_:inputs
            command=policy.act(state,refs);policy.observations=fn
            compare('action',policy.last,expected_raw)
            compare('target',np.asarray(command.joint_position_target_rad)[order],native.get_target_dof_pos(expected_raw))
            max_action=max(max_action,float(np.max(np.abs(expected_raw))))
            # Common measured action context prevents tiny FK rounding errors from recursive amplification.
            last=policy.last.copy();previous=reference.copy()
        synthetic=np.linspace(-20,20,29,dtype=np.float32)
        fn=policy.infer;policy.infer=lambda _: {'actions':synthetic[None]}
        result=policy.act(state,refs);policy.infer=fn
        compare('synthetic_target',np.asarray(result.joint_position_target_rad)[order],native.get_target_dof_pos(synthetic))
        compare('synthetic_history',policy.last,synthetic)
    limits={k:2e-4 for k in errors};limits['same_input_network']=0.;limits['synthetic_target']=1e-6;limits['synthetic_history']=0.;limits['fixed_initial_alignment']=1e-6
    report={'passed':all(errors[k]<=limits[k] for k in errors),'cases':160,'resets':4,'max_abs_error':errors,'limits':limits,'max_abs_native_action':max_action,'source_commit':rt['source_commit'],'checkpoint_sha256':rt['weights_sha256'],'scope':'Pinned original TeleopIT VelCmd observation builder, reference finite differences, history reset, CPU ONNX inference, raw action history and clipped motor targets. Shared canonical robot FK; shuffled SDK joints. Offline playback default smoothing=1. Fixed initial XY/yaw transform is identity under benchmark reference initialization. Hardware, retargeting and online transport are outside this audit.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']
if __name__=='__main__':main()
