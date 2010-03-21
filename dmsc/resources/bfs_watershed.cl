#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// #include <common_funcs.cl>

const sampler_t cell_own_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

void write_to_owner_image(short2 c,short2 data, __write_only image2d_t cell_own_image)
{
  int2 imgcrd;

  imgcrd.x = c.y;
  imgcrd.y = c.x;

  int4 data_val;

  data_val.x = data.x;
  data_val.y = data.y;
  data_val.z = 0;
  data_val.w = 0;

  write_imagei(cell_own_image, imgcrd,data_val);
}

__kernel void dobfs_markowner_extrema_init(
__read_only image2d_t  cell_fg_img,
__write_only image2d_t cell_own_image_out,
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
  
  uint flag = get_cell_flag(c,cell_fg_img);

  short2 own;
  
  if(is_cell_critical(flag) == 1)
  {
    own.x = c.x ;
    own.y = c.y ;
  }
  else
  {
    own.x = -1;
    own.y = -1;
  }
  
  write_to_owner_image(c,own,cell_own_image_out);
}

__kernel void dobfs_markowner_extrema(
__read_only image2d_t  cell_fg_img,
__read_only image2d_t  cell_pr_img,
__read_only image2d_t  cell_own_image_in,
__write_only image2d_t cell_own_image_out,
__global unsigned int * g_changed,
const short2 ext_bl,
const short2 ext_tr
)
{
  short2 cg,bb_ext_sz,cl;

  bb_ext_sz.x = ext_tr.x-ext_bl.x;
  bb_ext_sz.y = ext_tr.y-ext_bl.y;

  cg.x = get_global_id(0);
  cg.y = get_global_id(1);
  
  cl.x = get_local_id(0);
  cl.y = get_local_id(1);

  __local short2 cell_own_shr[BLOCK_DIM_X+2][BLOCK_DIM_Y+2];
  
//   first global access  
  int2 imgcrd; int4 data;
  
  short2 p;
  uint flag;
  
  data.x= 0 ;data.y = 0;data.z =0;data.w=0;
  
  imgcrd.y = cg.x;
  imgcrd.x = cg.y;
  
  if(cg.x <= bb_ext_sz.x && cg.y<=bb_ext_sz.y)
    data = read_imagei(cell_own_image_in, cell_own_sampler, imgcrd);

  cell_own_shr[cl.x+1][cl.y+1].x = data.x;
  cell_own_shr[cl.x+1][cl.y+1].y = data.y;    

  if(cg.x <= bb_ext_sz.x && cg.y<=bb_ext_sz.y)
    data = read_imagei(cell_pr_img, cell_pr_sampler, imgcrd);

  p.x = data.x - ext_bl.x;
  p.y = data.y - ext_bl.y;

  if(cg.x <= bb_ext_sz.x && cg.y<=bb_ext_sz.y)
    data = read_imagei(cell_fg_img, cell_fg_sampler, imgcrd);

  flag = data.x;
  
//    the border guys need to read the side.... for this to work makesure the grid is atleast 6x6
  int xadd = 0;
  int yadd = 0;
  
  xadd = (cl.x == 0)?-1:xadd;
  yadd = (cl.y == 0)?-1:yadd;
  
  xadd = (cl.x + 1 == get_local_size(0))?1:xadd;
  yadd = (cl.y + 1 == get_local_size(1))?1:yadd;
  
  xadd = ((cl.x == 2 && cl.y == 1)||(cl.x == 1 && cl.y == 2))?-2:xadd;
  yadd = ((cl.x == 2 && cl.y == 1)||(cl.x == 1 && cl.y == 2))?-2:yadd;
  
  xadd = ((cl.x == get_local_size(0) - 3 && cl.y == 1)||(cl.x == get_local_size(0) - 2 && cl.y == 2))?2:xadd;  
  yadd = ((cl.x == get_local_size(0) - 3 && cl.y == 1)||(cl.x == get_local_size(0) - 2 && cl.y == 2))?-2:yadd;
  
  xadd = ((cl.x == get_local_size(0) - 3 && cl.y == get_local_size(1) - 2)||(cl.x == get_local_size(0) - 2 && cl.y == get_local_size(1) - 3))?2:xadd;  
  yadd = ((cl.x == get_local_size(0) - 3 && cl.y == get_local_size(1) - 2)||(cl.x == get_local_size(0) - 2 && cl.y == get_local_size(1) - 3))?2:yadd;  
  
  xadd = ((cl.x == 2 && cl.y == get_local_size(1) - 2)||(cl.x == 1 && cl.y == get_local_size(1) - 3))?-2:xadd;  
  yadd = ((cl.x == 2 && cl.y == get_local_size(1) - 2)||(cl.x == 1 && cl.y == get_local_size(1) - 3))?2:yadd; 
  
//   second global access

  imgcrd.y = cg.x + xadd;
  imgcrd.x = cg.y + yadd;

  if(cg.x +xadd <= bb_ext_sz.x && cg.y+yadd <=bb_ext_sz.y)
    data = read_imagei(cell_own_image_in, cell_own_sampler, imgcrd);

  cell_own_shr[cl.x + 1 + xadd][cl.y + 1 + yadd].x = data.x;
  cell_own_shr[cl.x + 1 + xadd][cl.y + 1 + yadd].y = data.y;
  
  int d = get_cell_dim(cg);
  
  short2 inf_cell;int inf_valid = 0;
  
  int pd = get_cell_dim(p);
  
  short2 cets[2];
  
  if(cg.x <= bb_ext_sz.x && cg.y<=bb_ext_sz.y &&
     is_cell_critical(flag) == 0 &&
     is_cell_paired(flag) == 1)
  {     
    if(d == 0 || d == 2)
    {
      inf_cell.x = p.x;
      inf_cell.y = p.y;
    }
    
    if(d == 1)
    {           
      if(pd == 0) 
        get_cell_facets(cg,cets);
      else 
        get_cell_cofacets(cg,cets);
    
      int inf_idx = 0;
      
      if(cets[0].x == p.x && cets[0].y == p.y)
        inf_idx = 1;

      inf_cell.x = cets[inf_idx].x;
      inf_cell.y = cets[inf_idx].y;     
    }
    if(inf_cell.x <= bb_ext_sz.x && inf_cell.y<=bb_ext_sz.y)
      inf_valid = 1;
  }
  
  __local int  l_changed;   
  int need_global_update = 0;  
  
  l_changed = 1;
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  while(l_changed == 1)
  {         
    short2 own;
    own.x = cell_own_shr[cl.x+1][cl.y+1].x;
    own.y = cell_own_shr[cl.x+1][cl.y+1].y;    
    
    int need_local_update = 0;
    
    if(own.x == -1 && own.y == -1 && inf_valid == 1)
    {          
      short2 inf_cell_loc;
      inf_cell_loc.x = inf_cell.x -(cg.x - cl.x);
      inf_cell_loc.y = inf_cell.y -(cg.y - cl.y);
      
      short2 inf_cell_own;
      inf_cell_own.x = cell_own_shr[inf_cell_loc.x+1][inf_cell_loc.y+1].x;
      inf_cell_own.y = cell_own_shr[inf_cell_loc.x+1][inf_cell_loc.y+1].y;
      
      if(inf_cell_own.x != -1 && inf_cell_own.y != -1)
      {
        own.x = inf_cell_own.x;
        own.y = inf_cell_own.y;
        need_local_update = 1;
      }      
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    l_changed = 0;   
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(need_local_update == 1)
    {
      cell_own_shr[cl.x+1][cl.y+1].x = own.x;
      cell_own_shr[cl.x+1][cl.y+1].y = own.y;
      
      need_global_update = 1;

      l_changed = 1;
    }    
    barrier(CLK_LOCAL_MEM_FENCE);
  }  
  
  if(need_global_update == 1)
    g_changed[0] = 1;

  short2 own;
  own.x = cell_own_shr[cl.x+1][cl.y+1].x;
  own.y = cell_own_shr[cl.x+1][cl.y+1].y;  
  
  if(cg.x <= bb_ext_sz.x && cg.y<=bb_ext_sz.y )
    write_to_owner_image(cg,own,cell_own_image_out);
}