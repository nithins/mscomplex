////////////////////////////////////////////////////////////////////////////////

// Gradient assignment computation kernel
//

#define cell_fn_t    float
#define cell_coord_t short
#define cell_flag_t  uchar
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16


const sampler_t vert_fn_sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
const sampler_t cell_own_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
const sampler_t cell_fg_sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
const sampler_t cell_pr_sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

short get_cell_dim(const short2 c)
{
  int dim = (c.x&0x01) + (c.y&0x01);
  return dim;
}

short get_cell_facets(const short2 c,short2 *f)
{
  if(get_cell_dim(c) == 0)
  {
    return 0;
  }
  
  if(get_cell_dim(c) == 1)
  {
    f[0].x = c.x + c.x%2;
    f[0].y = c.y + c.y%2;
    
    f[1].x = c.x - c.x%2;
    f[1].y = c.y - c.y%2;
    return 2;
  }
  
  if(get_cell_dim(c) == 2)
  {
    f[0].x = c.x + 1;
    f[0].y = c.y;
    
    f[1].x = c.x - 1;
    f[1].y = c.y;
    
    f[2].x = c.x;
    f[2].y = c.y + 1;
    
    f[3].x = c.x;
    f[3].y = c.y - 1;
    return 4;
  }
  
  return 0;
}

short get_cell_cofacets(const short2 c,short2 *cf)
{
  if(get_cell_dim(c) == 0)
  {
    cf[0].x = c.x + 1;
    cf[0].y = c.y;
    
    cf[1].x = c.x - 1;
    cf[1].y = c.y;
    
    cf[2].x = c.x;
    cf[2].y = c.y + 1;
    
    cf[3].x = c.x;
    cf[3].y = c.y - 1;
    return 4;
  }
  
  if(get_cell_dim(c) == 1)
  {
    cf[0].x = c.x + c.y%2;
    cf[0].y = c.y + c.x%2;
    
    cf[1].x = c.x - c.y%2;
    cf[1].y = c.y - c.x%2;
    return 2;
  }
  
  if(get_cell_dim(c) == 2)
  {
    return 0;
  }
  
  return 0;
}

short get_cell_points(const short2 c,short2 *p)
{
  if(get_cell_dim(c) == 0)
  {
    p[0] = c;
    return 1;
  }
  
  if(get_cell_dim(c) == 1)
  {
    p[0].x = c.x + c.x%2;
    p[0].y = c.y + c.y%2;
    
    p[1].x = c.x - c.x%2;
    p[1].y = c.y - c.y%2;
    return 2;
  }
  
  if(get_cell_dim(c) == 2)
  {
    p[0].x = c.x + 1;
    p[0].y = c.y + 1;
    
    p[1].x = c.x - 1;
    p[1].y = c.y + 1;
    
    p[2].x = c.x - 1;
    p[2].y = c.y - 1;
    
    p[3].x = c.x + 1;
    p[3].y = c.y - 1;
    return 4;
  }  
  
  return 0;
}


cell_fn_t get_cell_fn(short2 c,__read_only image2d_t vert_fn_img)
{  
  int2 imgcrd;
  
  imgcrd.y = c.x/2;
  imgcrd.x = c.y/2;
  
  float4 col = read_imagef(vert_fn_img, vert_fn_sampler, imgcrd);
  
  return col[0];     
  
  //return sin(0.0f+0.125f*c.x + 0.5f)*sin(0.0f+0.125f*c.y + 0.5f);
}

int comparePoints(short2 c1,short2 c2,__read_only image2d_t vert_fn_img)
{    
  cell_fn_t f1 = get_cell_fn(c1, vert_fn_img);
  cell_fn_t f2 = get_cell_fn(c2, vert_fn_img);    
    
  if(f1 < f2 ) return 1;
  if(f2 < f1 ) return 0;
  
  if(c1.x < c2.x) return 1;
  if(c2.x < c1.x) return 0;
  
  if(c1.y < c2.y) return 1;   
  return 0;
}

int max_pt_idx(short2 *pts,int num_pts,int offset,__read_only image2d_t vert_fn_img)
{
  int pt_max_idx = offset;
  
  for(int i = offset+1 ; i < num_pts;i++)
    if(comparePoints(pts[pt_max_idx],pts[i], vert_fn_img) == 1)
      pt_max_idx = i;
    
  return pt_max_idx;  
}

int ins_sort_pts(short2 *pts,int num_pts,__read_only image2d_t vert_fn_img)
{  
    for(int i = 0 ; i < num_pts;i++)
    {      
      int swp_idx   = max_pt_idx(pts,num_pts,i,vert_fn_img);            
      short2 temp   = pts[i];
      pts[i]        = pts[swp_idx];
      pts[swp_idx]  = temp;
    }
}

int compareCells(short2 c1,short2 c2,__read_only image2d_t vert_fn_img)
{
  
  short2 pt1[4],pt2[4];
  
  int pt1_ct = get_cell_points(c1,pt1);
  int pt2_ct = get_cell_points(c2,pt2);      
  
  ins_sort_pts(pt1,pt1_ct, vert_fn_img);
  ins_sort_pts(pt2,pt2_ct, vert_fn_img);
  
  int num_lex_comp = min(pt1_ct,pt2_ct);
  
  for(int i = 0 ;i < num_lex_comp;++i)
  {
    if(comparePoints(pt1[i],pt2[i], vert_fn_img) == 1)
      return 1;
    
    if(comparePoints(pt2[i],pt1[i], vert_fn_img) == 1)
      return 0;
  }
  
  if (pt1_ct < pt2_ct) return 1;  
  
  return 0;  
}

__kernel void assign_gradient
( __read_only  image2d_t  vert_fn_img,
  __write_only image2d_t  cell_pr_img,
  __write_only image2d_t  cell_fg_img,
   const cell_coord_t x_min,
   const cell_coord_t x_max,
   const cell_coord_t y_min,   
   const cell_coord_t y_max)
   
{
    short2 c,bb_sz;
    
    bb_sz.x = x_max-x_min;
    bb_sz.y = y_max-y_min;
  
    c.x = get_global_id(0);
    c.y = get_global_id(1);
  
   
   if(c.x > bb_sz.x || c.y > bb_sz.y)
     return;
   
   int cf_usable[4];
   
   short2 cf[4];
   
   int cf_ct = get_cell_cofacets(c,cf);
   
   for( int i = 0 ; i < cf_ct;++i)
   {
     cf_usable[i]  = 1;
     cf_usable[i] &= (cf[i].x >= 0);
     cf_usable[i] &= (cf[i].y >= 0);     
     cf_usable[i] &= (cf[i].x <= x_max-x_min);
     cf_usable[i] &= (cf[i].y <= y_max-y_min);
     
     if(cf_usable[i] == 0 )
       continue;
     
     short2 f[4];
     
     int f_ct = get_cell_facets(cf[i],f);    
     
     for( int j = 0 ; j < f_ct;++j)
     {       
       if(compareCells(c,f[j], vert_fn_img) == 1)
	 cf_usable[i] = 0; 
     }
   }
   
   short2 p;
   int is_paired = 0;
   
   for( int i = 0 ; i < cf_ct;++i)
   {
     if(cf_usable[i] == 1) 
     {
       if(is_paired == 0 )
       {
	p = cf[i];
	is_paired = 1;
       }
       else
       {
	 if(compareCells(cf[i],p,vert_fn_img) == 1)
	   p = cf[i];
       }
     } 
   }  

   
   int2 ccoord;
   ccoord.y = c.x;
   ccoord.x = c.y;
   
   if(is_paired == 1)
   {
     int4 pr;
     pr.x = p.x + x_min;
     pr.y = p.y + y_min;
     pr.z = 0;
     pr.w = 0;
     
     write_imagei(cell_pr_img,ccoord,pr);
     
     uint4 fg;
     fg.x = 1;
     fg.y = 0;
     fg.z = 0;
     fg.w = 0;
     
     write_imageui(cell_fg_img,ccoord,fg);
   }
   else
   {
     uint4 fg;
     fg.x = 2;
     fg.y = 0;
     fg.z = 0;
     fg.w = 0;
     
     write_imageui(cell_fg_img,ccoord,fg);
   }   
}
__kernel void complete_pairings
(__read_only  image2d_t  cell_pr_img,
 __write_only image2d_t  cell_pr_img_out,
 __read_only  image2d_t  cell_fg_img,
 __write_only image2d_t  cell_fg_img_out,
   const cell_coord_t x_min,
   const cell_coord_t x_max,
   const cell_coord_t y_min,
   const cell_coord_t y_max)
{
    short2 c,bb_sz;

    bb_sz.x = x_max-x_min;
    bb_sz.y = y_max-y_min;

    c.x = get_global_id(0);
    c.y = get_global_id(1);


   if(c.x > bb_sz.x || c.y > bb_sz.y)
     return;
   
   int2 ccoord;
   ccoord.y = c.x;
   ccoord.x = c.y;
   
   uint4 cflgs_arr = read_imagei(cell_fg_img, cell_fg_sampler, ccoord);
   
   cell_flag_t cflgs = cflgs_arr[0] ;

   if(cflgs == 2)
   {
    short2 f[4];

    int f_ct = get_cell_facets(c,f);

    short2 p;
    int is_paired = 0;

    for( int i = 0 ; i < f_ct;++i)
    {
      int2 imgcrd;
      imgcrd.y = f[i].x;
      imgcrd.x = f[i].y;

      int4 fp = read_imagei(cell_pr_img, cell_pr_sampler, imgcrd);

      if(fp.x == c.x + x_min && fp.y == c.y + y_min)
      {
       p = f[i];
       is_paired = 1;
       break;
      }
    }

    if(is_paired == 1)
    {
     int4 pr;
     pr.x = p.x + x_min;
     pr.y = p.y + y_min;
     pr.z = 0;
     pr.w = 0;
     
     write_imagei(cell_pr_img_out,ccoord,pr);
     
     uint4 fg;
     fg.x = 1;
     fg.y = 0;
     fg.z = 0;
     fg.w = 0;
     
     write_imageui(cell_fg_img_out,ccoord,fg);
      
    }
  }
}

int is_cell_critical(uchar flag)
{  
   return (flag&0x00000002)?1:0;;  
}

__kernel void dobfs_markowner_extrema_init
(__read_only image2d_t  cell_fg_img,
 __write_only image2d_t cell_own_image_out,
 const cell_coord_t x_min,
 const cell_coord_t x_max,
 const cell_coord_t y_min,
 const cell_coord_t y_max)
 {
  short2 bb_sz;

  bb_sz.x = x_max-x_min;
  bb_sz.y = y_max-y_min;  
  
  short2 c;
  
  c.x = get_global_id(0);
  c.y = get_global_id(1);
  
  if(c.x > bb_sz.x || c.y > bb_sz.y)
    return;
  
  int2 imgcrd; int4 data;
  
  data.x= 0 ;data.y = 0;data.z =0;data.w=0;
  
  imgcrd.y = c.x;
  imgcrd.x = c.y;  

  data = read_imagei(cell_fg_img, cell_fg_sampler, imgcrd);
  
  uchar flag = data.x; 
  
  if(is_cell_critical(flag) == 1)
  {
    data.x = c.x;
    data.y = c.y;
  }
  else
  {
    data.x = -1;
    data.y = -1;
  }
  
  write_imagei(cell_own_image_out, imgcrd,data); 

 }

__kernel void dobfs_markowner_extrema
(__read_only image2d_t  cell_fg_img, 
 __read_only image2d_t  cell_pr_img,
 __read_only image2d_t  cell_own_image_in,
 __write_only image2d_t cell_own_image_out,
 const cell_coord_t x_min,
 const cell_coord_t x_max,
 const cell_coord_t y_min,
 const cell_coord_t y_max,
 __global unsigned int * g_changed
 )
{
  
  short2 bb_sz;

  bb_sz.x = x_max-x_min;
  bb_sz.y = y_max-y_min;  

  __local short2 cell_own_shr[BLOCK_DIM_X+2][BLOCK_DIM_Y+2];  
  short2 cg,cl;
  
  cg.x = get_global_id(0);
  cg.y = get_global_id(1);
  
  cl.x = get_local_id(0);
  cl.y = get_local_id(1);  
  
//   first global access  
  int2 imgcrd; int4 data;
  
  short2 p;
  uchar flag;
  
  data.x= 0 ;data.y = 0;data.z =0;data.w=0;
  
  imgcrd.y = cg.x;
  imgcrd.x = cg.y;
  
  if(cg.x <= bb_sz.x && cg.y<=bb_sz.y)
    data = read_imagei(cell_own_image_in, cell_own_sampler, imgcrd);

  cell_own_shr[cl.x+1][cl.y+1].x = data.x;
  cell_own_shr[cl.x+1][cl.y+1].y = data.y;    

  if(cg.x <= bb_sz.x && cg.y<=bb_sz.y)
    data = read_imagei(cell_pr_img, cell_pr_sampler, imgcrd);

  p.x = data.x;
  p.y = data.y;  

  if(cg.x <= bb_sz.x && cg.y<=bb_sz.y)
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

  if(cg.x +xadd <= bb_sz.x && cg.y+yadd <=bb_sz.y)
    data = read_imagei(cell_own_image_in, cell_own_sampler, imgcrd);

  cell_own_shr[cl.x + 1 + xadd][cl.y + 1 + yadd].x = data.x;
  cell_own_shr[cl.x + 1 + xadd][cl.y + 1 + yadd].y = data.y;
  
  int d = get_cell_dim(cg);
  
  short2 inf_cell;int inf_valid = 0;
  
  int pd = get_cell_dim(p);
  
  short2 cets[2];
  
  if(cg.x <= bb_sz.x && cg.y<=bb_sz.y && is_cell_critical(flag) == 0)
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
    if(inf_cell.x <= bb_sz.x && inf_cell.y<=bb_sz.y)
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
  
  data.z=0;data.w =0;
  
  imgcrd.y = cg.x;
  imgcrd.x = cg.y;       
  
  data.y = cell_own_shr[cl.x+1][cl.y+1].x;
  data.x = cell_own_shr[cl.x+1][cl.y+1].y;
  
  if(cg.x <= bb_sz.x && cg.y<=bb_sz.y /*data.x != -1 && data.y != -1*/)
  {
    write_imagei(cell_own_image_out, imgcrd,data);
  }  
}