#define cell_coord_t short
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_DIM 128

const sampler_t cell_fg_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

int is_cell_critical(int2 c, __read_only image2d_t cell_fg_img)
{
   uint4 val = read_imageui(cell_fg_img, cell_fg_sampler, c);
   
   int data = (val.x&0x00000002)?1:0;
   
   return data;  
}

__kernel void collate_cps_initcount
(__read_only   image2d_t  cell_fg_img, 
 __global unsigned int* critpt_ct,
 const cell_coord_t x_min,
 const cell_coord_t x_max,
 const cell_coord_t y_min,
 const cell_coord_t y_max)
 {
   int2 c,bb_sz;
   
   bb_sz.x = x_max-x_min;
   bb_sz.y = y_max-y_min;     

   c.x = get_global_id(0);
   c.y = get_global_id(1);   
   
   if(c.x <= bb_sz.x && c.y <= bb_sz.y)
     critpt_ct[(bb_sz.x+1)*c.y + c.x]= is_cell_critical(c,cell_fg_img);
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