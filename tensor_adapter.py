"""TeleopIT batched reference finite differences and ten-frame history."""
from tensor_common import TensorPD,normalize,conjugate,rotate,multiply,torch


def axis(angle,index):
    values=[torch.zeros_like(angle) for _ in range(3)];values[index]=(angle/2).sin()
    return torch.stack((*values,(angle/2).cos()),dim=-1)


def wmul(a,b):
    w,x,y,z=a.unbind(-1);v,i,j,k=b.unbind(-1)
    return torch.stack((w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v),dim=-1)


def winv(q):
    return q*torch.tensor([1.,-1.,-1.,-1.],device=q.device)


def wrotate(q,v):
    return wmul(wmul(q,torch.cat((torch.zeros_like(v[...,:1]),v),dim=-1)),winv(q))[...,1:]


def rotation6(q):
    w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*y+w*z),1-2*(x*x+z*z),2*(x*z-w*y),2*(y*z+w*x)),dim=-1)


class TensorTeleopIT(TensorPD):
    def __init__(self,root,runtime,joint_names,slots,device,converted_weights):
        super().__init__(root,runtime,joint_names,slots,device,converted_weights)
        self.default=self.default.float();self.scale=self.scale.float()
        self.waist=[joint_names.index('waist_'+a+'_joint') for a in ('yaw','roll','pitch')]
        self.previous_q=torch.zeros((slots,29),device=device);self.previous_p=torch.zeros((slots,3),device=device);self.previous_r=torch.zeros((slots,4),device=device)
        self.history=torch.zeros((slots,10,167),device=device)

    def torso(self,rootq,joints,rootp=None):
        q=normalize(rootq.double());yaw=axis(joints[...,self.waist[0]].double(),2);q=multiply(q,yaw)
        p=None
        if rootp is not None:p=rootp.double()+rotate(q,torch.tensor([-.0039635,0.,.044],device=self.device,dtype=torch.float64))
        q=multiply(multiply(q,axis(joints[...,self.waist[1]].double(),0)),axis(joints[...,self.waist[2]].double(),1))
        return normalize(q).float()[...,[3,0,1,2]],p.float() if p is not None else None

    def observations(self,state,reference):
        aq,_=self.torso(state.root_orientation_xyzw,state.joint_position_rad)
        rq,rp=self.torso(reference.root_orientation_xyzw[:,0],reference.joint_position_rad[:,0],reference.root_position_world_m[:,0])
        baseq=state.root_orientation_xyzw.float()[:,[3,0,1,2]]
        q=state.joint_position_rad[:,self.order].float();qd=state.joint_velocity_rad_s[:,self.order].float()
        tq=reference.joint_position_rad[:,0,self.order].float();dt=torch.tensor(self.runtime['control_period_s'],device=self.device,dtype=torch.float32)
        tv=(tq-self.previous_q)*50;lv=(rp-self.previous_p)/dt;delta=wmul(rq,winv(self.previous_r))
        delta=torch.where(delta[:,:1]<0,-delta,delta)
        half=delta[:,0].double().clamp(-1,1).acos().float();sine=half.sin()
        av=delta[:,1:]/sine.clamp_min(1e-30)[:,None]*2*half[:,None]/dt
        av=torch.where((sine>1e-6)[:,None],av,0.)
        tv=torch.where(self.fresh[:,None],0.,tv);lv=torch.where(self.fresh[:,None],0.,lv);av=torch.where(self.fresh[:,None],0.,av)
        self.previous_q.copy_(tq);self.previous_p.copy_(rp);self.previous_r.copy_(rq)
        g=torch.tensor([0.,0.,-1.],device=self.device).expand(self.slots,3)
        omega=wrotate(winv(baseq),state.root_angular_velocity_world_rad_s.float())
        obs=torch.cat((tq,tv,rotation6(wmul(winv(aq),rq)),omega,q-self.default,qd,self.last,wrotate(winv(baseq),g),wrotate(winv(aq),lv),wrotate(winv(aq),av),wrotate(winv(rq),g),rp[:,2:3]),dim=-1)
        normal=torch.cat((self.history[:,1:],obs[:,None]),dim=1)
        self.history.copy_(torch.where(self.fresh[:,None,None],obs[:,None],normal));self.fresh.zero_()
        return {'obs':obs,'obs_history':self.history.clone()}


ADAPTER_CLASS='TensorTeleopIT'
NATIVE_CLASS='TeleopIT'
