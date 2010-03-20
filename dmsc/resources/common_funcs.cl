#define cell_fn_t    float
#define cell_coord_t short
#define cell_flag_t  uchar

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

int is_cell_critical(unsigned int flag)
{  
   return (flag&0x02)?1:0;  
}

int is_cell_paired(unsigned int flag)
{
   return (flag&0x01)?1:0;
}

unsigned int  get_cell_flag(short2 c, __read_only image2d_t cell_fg_img)
{  
  int2 imgcrd;
  
  imgcrd.x = c.y;
  imgcrd.y = c.x;
  
  uint4 val = read_imageui(cell_fg_img, cell_fg_sampler, imgcrd);
  
  return val.x;
}