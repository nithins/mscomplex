////////////////////////////////////////////////////////////////////////////////

// Gradient assignment computation kernel
//

#define cell_fn_t    float
#define cell_coord_t short
#define cell_flag_t  uchar

short get_cell_dim(const short2 c)
{
  int dim = (c.x%2) + (c.y%2);
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


cell_fn_t get_cell_fn(short2 c,short2 bb_sz,
		      __global cell_fn_t *cell_fns)
{  
  
  c.x /=2 ;
  c.y /=2 ; 
  
  int fn_pos = c.x *((bb_sz.y/2)+1) + c.y;
  
  return cell_fns[fn_pos];  
  
  // return sin(0.0f+0.125f*c.x + 0.5f)*sin(0.0f+0.125f*c.y + 0.5f);
  
  
}

int comparePoints(short2 c1,short2 c2,short2 bb_sz,
		  __global cell_fn_t *cell_fns)
{    
  cell_fn_t f1 = get_cell_fn(c1,bb_sz,cell_fns);
  cell_fn_t f2 = get_cell_fn(c2,bb_sz,cell_fns);    

    
  if(f1 < f2 ) return 1;
  if(f2 < f1 ) return 0;
  
  if(c1.x < c2.x) return 1;
  if(c2.x < c1.x) return 0;
  
  if(c1.y < c2.y) return 1;   
  return 0;
}

int max_pt_idx(short2 *pts,int num_pts,short2 bb_sz,
	       __global cell_fn_t *cell_fns)
{
  int pt_max_idx = 0;
  
  for(int i = 1 ; i < num_pts;i++)
    if(comparePoints(pts[pt_max_idx],pts[i],bb_sz,cell_fns) == 1)
      pt_max_idx = i;
    
  return pt_max_idx;  
}

void ins_sort_pts(short2 *pts,int num_pts,short2 bb_sz,
	          __global cell_fn_t *cell_fns)
{
    for(int i = 0 ; i < num_pts;i++)
    {
      int swp_idx   = max_pt_idx(pts+i,num_pts-i,bb_sz,cell_fns) + i;      
      short2 temp   = pts[i];
      pts[i]        = pts[swp_idx];
      pts[swp_idx]  = temp;
    }
}

int compareCells(short2 c1,short2 c2,short2 bb_sz,
		  __global cell_fn_t *cell_fns)
{
  
  short2 pt1[4],pt2[4];
  
  int pt1_ct = get_cell_points(c1,pt1);
  int pt2_ct = get_cell_points(c2,pt2);      
  
  ins_sort_pts(pt1,pt1_ct,bb_sz,cell_fns);
  ins_sort_pts(pt2,pt2_ct,bb_sz,cell_fns);
  
  int num_lex_comp = min(pt1_ct,pt2_ct);
  
  for(int i = 0 ;i < num_lex_comp;++i)
  {
    if(comparePoints(pt1[i],pt2[i],bb_sz,cell_fns) == 1)
      return 1;
    
    if(comparePoints(pt2[i],pt1[i],bb_sz,cell_fns) == 1)
      return 0;
  }
  
  if (pt1_ct < pt2_ct) return 1;  
  
  return 0;  
}
__kernel void assign_gradient
( __global cell_fn_t    *cell_fns,      
  __global cell_coord_t *cell_pairs,
  __global cell_flag_t  *cell_flags,     
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
       if(compareCells(c,f[j],bb_sz,cell_fns) == 1)
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
	 if(compareCells(cf[i],p,bb_sz,cell_fns) == 1)
	   p = cf[i];
       }	 
     } 
   }
   
   uint cell_op_idx = (c.x)*(y_max - y_min+1) + (c.y);
   
   if(is_paired == 1)
   {   
    cell_pairs[cell_op_idx*2+0] = x_min+p.x;
    cell_pairs[cell_op_idx*2+1] = y_min+p.y;
    
    cell_flags[cell_op_idx] = 1;
   }
   else
   {
     cell_flags[cell_op_idx] = 2;
   }   
}                              


