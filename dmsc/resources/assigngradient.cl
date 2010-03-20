////////////////////////////////////////////////////////////////////////////////

// Gradient assignment computation kernel
//

// #include <common_funcs.cl>


const sampler_t vert_fn_sampler  = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

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
   const short2 x_int_range,
   const short2 y_int_range,
   const short2 x_ext_range,
   const short2 y_ext_range
)
   
{  
  short2 c,bb_ext_sz,bb_int_sz;
  
  bb_ext_sz.x = x_ext_range[1]-x_ext_range[0];
  bb_ext_sz.y = y_ext_range[1]-y_ext_range[0];    
  
  bb_int_sz.x = x_int_range[1]-x_int_range[0];
  bb_int_sz.y = y_int_range[1]-y_int_range[0];    
  
  if(get_global_id(0) > bb_int_sz.x || 
     get_global_id(1) > bb_int_sz.y)
   return;    
  
  c.x = get_global_id(0) + x_int_range[0] - x_ext_range[0];
  c.y = get_global_id(1) + y_int_range[0] - y_ext_range[0] ;
  
  int cf_usable[4];
  
  short2 cf[4];
  
  int cf_ct = get_cell_cofacets(c,cf);
  
  for( int i = 0 ; i < cf_ct;++i)
  {
    cf_usable[i]  = 1;
    cf_usable[i] &= (cf[i].x >= 0);
    cf_usable[i] &= (cf[i].y >= 0);     
    cf_usable[i] &= (cf[i].x <= bb_ext_sz.x);
    cf_usable[i] &= (cf[i].y <= bb_ext_sz.y);
    
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
    pr.x = p.x + x_ext_range[0];
    pr.y = p.y + y_ext_range[0];
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
   const short2 x_int_range,
   const short2 y_int_range,
   const short2 x_ext_range,
   const short2 y_ext_range
)
   
{  
  short2 c,bb_ext_sz,bb_int_sz;
  
  bb_ext_sz.x = x_ext_range[1]-x_ext_range[0];
  bb_ext_sz.y = y_ext_range[1]-y_ext_range[0];    
  
  bb_int_sz.x = x_int_range[1]-x_int_range[0];
  bb_int_sz.y = y_int_range[1]-y_int_range[0];    
  
  if(get_global_id(0) > bb_ext_sz.x || 
     get_global_id(1) > bb_ext_sz.y)
   return;    
  
  c.x = get_global_id(0) - x_ext_range[0];
  c.y = get_global_id(1) - y_ext_range[0] ;
   
   int2 ccoord;
   ccoord.y = c.x;
   ccoord.x = c.y;
   
   uint4 cflgs_arr = read_imagei(cell_fg_img, cell_fg_sampler, ccoord);
   
   cell_flag_t cflgs = cflgs_arr[0] ;

   if(is_cell_critical(cflgs))
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

      if(fp.x == c.x + x_ext_range[0] && 
	 fp.y == c.y + y_ext_range[0])
      {
       p = f[i];
       is_paired = 1;
       break;
      }
    }

    if(is_paired == 1)
    {
     int4 pr;
     pr.x = p.x + x_ext_range[0];
     pr.y = p.y + y_ext_range[0];
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