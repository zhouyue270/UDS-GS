import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.models import resnet18,resnet34,resnet50,resnet101
import torch.nn.functional as F
import copy
from lib.models.gaussian_model import GaussianModel


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def get_embedder_dim1(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 1,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
pass


class lin_module(nn.Module):
    def __init__(self,
                d_in,
                d_out,
                dims,
                multires=0,
                act_fun=None,last_act_fun=None,weight_norm=False,weight_zero=False,weight_xavier=True):
        super().__init__()
        
        dims = [d_in] + dims + [d_out]
        self.num_layers = len(dims)
        if act_fun==None:
            self.act_fun = nn.Softplus(beta=100)
        else:
            self.act_fun=act_fun
        self.last_act_fun=last_act_fun
        
        for l in range(0, self.num_layers -1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_zero:
                torch.nn.init.normal_(lin.weight, 0.0,0.0)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            if weight_xavier:
                torch.nn.init.xavier_normal_(lin.weight)
                torch.nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin" + str(l), lin)

    def forward(self, inx):

        x = inx
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            
            if l==self.num_layers-2:
                if self.last_act_fun is not None:
                    x=self.last_act_fun(x)
            else:
                x = self.act_fun(x)
        return x




from collections import OrderedDict
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class OneConv(nn.Module):
    def __init__(self, in_channels, out_channels,not_act=False):
        super(OneConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if not_act:
            self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.conv(x)
    
class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self,model,return_layers):

        orig_return_layers=return_layers
        return_layers={k: v for k,v in return_layers.items()}
        layers=OrderedDict()
        #
        for name,module in model.named_children():
            layers[name]=module
            #
            if name in return_layers:
                del return_layers[name]
            #
            if not return_layers:
                break

        super(IntermediateLayerGetter,self).__init__(layers)
        self.return_layers=orig_return_layers

    def forward(self,x):
        out=OrderedDict()
        #
        for name,module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                #
                out_name=self.return_layers[name]

                out[out_name]=x
        return out
    
class Unet_model(nn.Module):
    def __init__(self,features_dim=32,backbone="resnet18",use_features_mask=False,use_independent_mask_branch=False):
        super().__init__()
        features_dim=features_dim

        #
        if backbone=="resnet18":
            resnet=resnet18(weights=torchvision.models.ResNet18_Weights)
        elif backbone=="resnet34":
            resnet = resnet34(weights=torchvision.models.ResNet34_Weights)
        elif backbone=="resnet50":
            resnet = resnet50(weights=torchvision.models.ResNet50_Weights)
        elif backbone=="resnet101":
            resnet = resnet101(weights=torchvision.models.ResNet101_Weights)
        #
        backbone=nn.Sequential(*list(resnet.children())[:-2])#-2 16 16  -3 32 32
        return_layers={'0': 'low','4': '64','5': '128','6': '256',"7":"512"}
        self.encoder=IntermediateLayerGetter(backbone,return_layers=return_layers)
        backbone_channels,self.backbone_channels = [ 64,64, 128, 256, 512],[ 64,64, 128, 256, 512]
        
        self.decoder=nn.ModuleList()
        for i in range(len(backbone_channels) - 2, -1, -1):
            self.decoder.append(
                nn.ConvTranspose2d(backbone_channels[i+1], backbone_channels[i], kernel_size=2, stride=2),
                # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                # nn.Conv2d(backbone_channels[i]*2, backbone_channels[i], kernel_size=3, stride=1,padding=1)
            )
            self.decoder.append(DoubleConv(backbone_channels[i]*2, backbone_channels[i]))
        
        self.final_conv = nn.Conv2d(backbone_channels[0], features_dim, kernel_size=1)
        
        self.use_features_mask=use_features_mask
        if use_features_mask:
            self.use_independent_mask_branch=use_independent_mask_branch
            if use_independent_mask_branch:
                print("unet:use_independent_mask_branch")
                self.mask_encoder=copy.deepcopy(self.encoder)
            
            self.mask_decoder=nn.Sequential(
                nn.ConvTranspose2d(backbone_channels[-1], backbone_channels[-2], kernel_size=2, stride=2),
                OneConv(backbone_channels[-2],backbone_channels[-2]),
                nn.ConvTranspose2d(backbone_channels[-2], backbone_channels[-3], kernel_size=2, stride=2),
                OneConv(backbone_channels[-3],backbone_channels[-3]),
                nn.ConvTranspose2d(backbone_channels[-3], backbone_channels[-4], kernel_size=2, stride=2),
                OneConv(backbone_channels[-4],backbone_channels[-4]),
                nn.Conv2d(backbone_channels[-4], 1, kernel_size=1),
                nn.Sigmoid()
            )

    
    def forward(self, inx,only_gloabal_features=False,eval_mode=False):
        out={}
        features=self.encoder(inx)
        x=features["512"]
        if self.use_features_mask:
            infm=features["512"]
            
            if self.use_independent_mask_branch:
                infm=self.mask_encoder(inx)["512"]
            out["mask"]=self.mask_decoder(infm)

        
        for i in range(0, len(self.decoder)-1, 2):
            x = self.decoder[i](x)
            x0=features[str(self.backbone_channels[-i//2-2])]
            x=torch.nn.functional.interpolate(x,size=(x0.shape[-2],x0.shape[-1]))
            x = torch.cat([x,x0] , dim=1)
            x = self.decoder[i+1](x)
        x = self.decoder[-2](x)
        x0=features[str("low")]
        x=torch.nn.functional.interpolate(x,size=(x0.shape[-2],x0.shape[-1]))
        x = torch.cat([x, x0], dim=1)
        x = self.decoder[-1](x)

        out["feature_maps"] = self.final_conv(x)
            
        return out


def project2d(pointcloud,world2camera,camera2image,box_coord,feature_map):
    xyz_world_h=torch.cat([pointcloud,torch.ones(size=(pointcloud.shape[0],1), dtype=pointcloud.dtype, device="cuda")],dim=1)
    xyz_camera=torch.mm(xyz_world_h,world2camera)[:,:3]       #N,3   
    xy_image0=torch.mm(xyz_camera,camera2image.transpose(0, 1))  #N,3  
    xy_image=torch.zeros((xy_image0.shape[0],2),device=xy_image0.device)
    xy_image[:,0]=xy_image0[:,0]/(xy_image0[:,2]+1e-2)
    xy_image[:,1]=xy_image0[:,1]/(xy_image0[:,2]+1e-2)    #N，3    

    mask_in_width=torch.logical_and(0<xy_image[:,0],xy_image[:,0]<box_coord[:,1])  #3,H,W
    mask_in_height=torch.logical_and(0<xy_image[:,1],xy_image[:,1]<box_coord[:,0])
    mask_in_image=torch.logical_and(mask_in_width, mask_in_height)
    mask_front_point=xy_image0[:,2]>0
    valid_point_mask=torch.logical_and(mask_in_image,mask_front_point)

    
    valid_pixel=xy_image[valid_point_mask][:,:2]     
    if box_coord.shape[0]>1:
        valid_box_coord=box_coord[valid_point_mask]
    else:
        valid_box_coord=box_coord
    valid_pixelx_normalized=valid_pixel[:,0]/(valid_box_coord[:,1]/2)-1
    valid_pixely_normalized=valid_pixel[:,1]/(valid_box_coord[:,0]/2)-1
    valid_pixel_normal=torch.stack((valid_pixelx_normalized,valid_pixely_normalized),dim=1)
    valid_pixel_normal=torch.unsqueeze(valid_pixel_normal,0)
    valid_pixel_normal=torch.unsqueeze(valid_pixel_normal,0)
    point_feature=F.grid_sample(feature_map,valid_pixel_normal,mode='bilinear', padding_mode='border').squeeze().T
    point_feature_all=torch.zeros(size=(pointcloud.shape[0],point_feature.shape[1]),dtype=pointcloud.dtype,device=pointcloud.device)
    
    point_feature_all[valid_point_mask]=point_feature

    return point_feature_all,valid_point_mask


def sample_from_feature_maps(feature_maps, pts,box_coord,coord_scale=1,combine_method="cat", mode='bilinear', padding_mode='border'):

    n_maps, C, H, W = feature_maps.shape

    #feature_maps = feature_maps.view(N*n_maps, C, H, W)
    coordinates=box_coord.permute(1,0,2)/coord_scale

    coordinates=coordinates.unsqueeze(1)# n_maps 1 M 2

    
    output_features = torch.nn.functional.grid_sample(feature_maps, coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(2,3,0,1).squeeze()
    if combine_method=="cat":
        output_features=output_features.reshape((output_features.shape[0],-1)) 
    elif combine_method=="sum":
        output_features=output_features.sum(dim=1)
    return output_features,coordinates


class Color_net(nn.Module):
    def __init__(self,
                fin_dim,
                pin_dim,
                view_dim,
                pfin_dim,
                en_dims,
                de_dims,
                multires,
                pre_compc=True,
                cde_dims=[48],
                use_pencoding=[True,False],
                weight_norm=False,
                weight_xavier=True,
                use_drop_out=True,
                use_decode_with_pos=False,
                ):
        super().__init__()
        self.pre_compc=pre_compc
        self.use_pencoding=use_pencoding
        self.embed_fns=[]
        self.cache_outd=None
        self.use_decode_with_pos=use_decode_with_pos
        if use_pencoding[0]:
            embed_fn, input_ch = get_embedder(multires[0]) # 63
            pin_dim = input_ch
            self.embed_fns.append(embed_fn)
        else:
            self.embed_fns.append(None)
            
        if use_pencoding[1]:
            embed_fn, input_ch = get_embedder(multires[1])
            view_dim = input_ch
            self.embed_fns.append(embed_fn)
        else:
            self.embed_fns.append(None)

            
        self.encoder=lin_module(fin_dim+pin_dim+pfin_dim,fin_dim,en_dims,multires[0],act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
        self.decoder=lin_module(75,fin_dim,de_dims,multires[0],act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
        if self.pre_compc:
            # view_dim=3
            self.color_decoder=lin_module(fin_dim+view_dim,3,cde_dims,multires[0],act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
            #self.color_decoder=lin_module(fin_dim+pin_dim,3,cde_dims,multires[0],act_fun=nn.ReLU(),weight_norm=weight_norm,weight_xavier=weight_xavier)
        self.use_drop_out=use_drop_out
        
        if use_drop_out:
            self.drop_outs=[nn.Dropout(0.1)]

                
    def forward(self, inp,inf,inpf,view_direction=None,inter_weight=1.0,store_cache=False):
        oinp=inp
        if self.use_drop_out:
            inpf=self.drop_outs[0](inpf)
        if  self.use_pencoding[0]:
            if self.use_decode_with_pos:
                oinp=inp.clone()
            inp = self.embed_fns[0](inp)

        if  self.use_pencoding[1]:
            view_direction = self.embed_fns[1](view_direction)
            #view_direction=self.embed_fn(view_direction)
        p_num=inf.shape[0]
        inf=inf.reshape([p_num,-1])
        
        inpf=inpf*inter_weight
        inx=torch.cat([inp,inpf,inf],dim=1)
        #inx=torch.cat([inp,inf],dim=1)
        oute= self.encoder(inx)
        outd=self.decoder(torch.cat([oute,inf],dim=1))
        if store_cache:
            self.cache_outd=outd
        else:
            self.cache_outd=None
            
        if self.pre_compc:
            if self.use_decode_with_pos:
                outc=self.color_decoder(torch.cat([outd,oinp],dim=1))
            else:
                outc=self.color_decoder(torch.cat([outd,view_direction],dim=1)) #view_direction
            return outc
        return outd.reshape([p_num,-1,3])

    def forward_cache(self, inp,view_direction=None):
        oinp=inp
        if  self.use_pencoding[0]:
            if self.use_decode_with_pos:
                oinp=inp.clone()
            inp = self.embed_fns[0](inp)
        if  self.use_pencoding[1]:
            view_direction = self.embed_fns[1](view_direction)
        p_num=inp.shape[0]
        if self.pre_compc:
            if self.use_decode_with_pos:
                outc=self.color_decoder(torch.cat([self.cache_outd,oinp],dim=1))
            else:
                outc=self.color_decoder(torch.cat([self.cache_outd,view_direction],dim=1)) #view_direction
            return outc
        return self.cache_outd.reshape([p_num,-1,3])
        

    

        





class WhloeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.featget = Unet_model()
        features_dim= 11
        self.colorNet = Color_net(48,3,3,features_dim,[128,75,64],[48,48],[10,0])
    

    def forward(self,pc:GaussianModel,viewpoint_camera):
        
        camera_center=viewpoint_camera.camera_center
        view_direction=(pc.get_xyz - camera_center.repeat(pc.get_xyz.shape[0], 1))
        view_direction=view_direction/(view_direction.norm(dim=1, keepdim=True)+1e-5)



        img=viewpoint_camera.original_image.cuda()
        feature_maps = self.featget(img.unsqueeze(0))['feature_maps']

        # box
        norm_xyz=(pc.get_xyz-pc.get_xyz.min(dim=0)[0])/(pc.get_xyz.max(dim=0)[0]-pc.get_xyz.min(dim=0)[0])
        norm_xyz=(norm_xyz-0.5)*2
        self.map_num = 4
        self.coord_scale = 1
        box_coord=torch.rand(size=(pc.get_xyz.shape[0],self.map_num,2),device="cuda")
        for i in range(self.map_num-1):
            rand_weight=torch.rand(size=(2,3),device="cuda")
            rand_weight=rand_weight/rand_weight.sum(dim=-1).unsqueeze(1)
            box_coord[:,i,:]=torch.einsum('bi,ni->nb', rand_weight, norm_xyz)*self.coord_scale
            # logging.info((f"rand sample coordinate weight: {rand_weight}"))
        # box_coord=box_coord
        self.box_coord=nn.Parameter(box_coord.requires_grad_(True))
        self.box_coord1=nn.Parameter(box_coord[:,-1,:].detach().requires_grad_(True))
        self.box_coord2=nn.Parameter(box_coord[:,:-1,:].detach().requires_grad_(False))


        box_coord1,box_coord2=self.box_coord1,self.box_coord2
        feature_maps=feature_maps.reshape(self.map_num,-1,feature_maps.shape[-2],feature_maps.shape[-1])
        _point_features0,self.map_pts_norm=sample_from_feature_maps(feature_maps[:self.map_num-1,...],pc.get_xyz,box_coord2,self.coord_scale)


        _point_features1,project_mask=project2d(pc.get_xyz,viewpoint_camera.world_view_transform,viewpoint_camera.K,box_coord1,feature_maps[-1,...].unsqueeze(0))
        _point_features=torch.cat([_point_features0,_point_features1],dim=1)
        features = torch.zeros(pc.get_xyz.shape[0], 3, 9).float().cuda()
        self._features_intrinsic = nn.Parameter(features[:,:,:].transpose(1, 2).contiguous().requires_grad_(True))
        features_dealed=self.colorNet(pc.get_xyz, self._features_intrinsic, _point_features,view_direction,\
                        store_cache=False)

        return features_dealed
        








if __name__=="__main__":
    inx = torch.rand((1,3,500,513))
    fcn_model=Unet_model()
    out=fcn_model(inx)
    pass

