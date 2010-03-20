#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_DIM 128

// #include <common_funcs.cl>

void write_crit_pt_idx_to_pr_image(short2 c, unsigned int idx,__write_only image2d_t  cell_pr_img)
{
  int4 data;
  
  int2 imgcrd;
  
  imgcrd.x = c.y;
  imgcrd.y = c.x;
  
  data.x = idx&0xff;
  data.y = (idx>>16)&0xff;
  
  data.z = 0; data.w = 0;
  
  write_imagei(cell_pr_img, imgcrd, data);
}

__kernel void collate_cps_initcount(
__read_only   image2d_t  cell_fg_img, 
__global unsigned int* critpt_ct,
const short2 ext_bl,
const short2 ext_tr
)
{
  short2 c,bb_ext_sz;

  bb_ext_sz.x = ext_tr.x-ext_bl.x;
  bb_ext_sz.y = ext_tr.y-ext_bl.y;

  if(get_global_id(0) > bb_ext_sz.x ||
     get_global_id(1) > bb_ext_sz.y)
   return;

  c.x = get_global_id(0);
  c.y = get_global_id(1);
  critpt_ct[c.y*(bb_ext_sz.x+1) + c.x]= is_cell_critical(get_cell_flag(c,cell_fg_img));
}
 
__kernel void collate_cps_writeids(
__read_only  image2d_t  cell_fg_img,
__write_only image2d_t  cell_pr_img,
__global unsigned int* critpt_idx,
__global short*        critpt_cellid,
const short2 ext_bl,
const short2 ext_tr
)
{
  short2 c,bb_ext_sz;

  bb_ext_sz.x = ext_tr.x-ext_bl.x;
  bb_ext_sz.y = ext_tr.y-ext_bl.y;

  if(get_global_id(0) > bb_ext_sz.x ||
     get_global_id(1) > bb_ext_sz.y)
   return;

  c.x = get_global_id(0);
  c.y = get_global_id(1);

  if(is_cell_critical(get_cell_flag(c,cell_fg_img)) == 1)
  {
    // write out cellid in the cellid array
    unsigned int idx = critpt_idx[c.y*(bb_ext_sz.x+1) + c.x];
    critpt_cellid[2*idx + 0] = c.x+ext_bl.x;
    critpt_cellid[2*idx + 1] = c.y+ext_bl.y;
    
    write_crit_pt_idx_to_pr_image(c,idx,cell_pr_img);
  }
}

__kernel void collate_cps_reduce(__global int*  critpt_ct,int n)
{
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);
    
    __local int sdata[BLOCK_DIM];

    sdata[tid] = (i < n) ? critpt_ct[i] : 0;
    if (i + get_local_size(0) < n) 
        sdata[tid] += critpt_ct[i+get_local_size(0)];  

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for(unsigned int s=get_local_size(0)/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem 
    if (tid == 0) critpt_ct[get_group_id(0)] = sdata[0];
}