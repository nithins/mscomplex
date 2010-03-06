#include <grid_dataset.h>

#include <discreteMorseAlgorithm.h>

#include <vector>

typedef GridDataset::cellid_t cellid_t;

GridDataset::GridDataset(const rect_t &r,const rect_t &e):
    m_rect(r),m_ext_rect(e)
{

  // TODO: assert that the given rect is of even size..
  //       since each vertex is in the even positions
  //  
}

void GridDataset::init()
{
  rect_size_t   s = m_ext_rect.size();  

  m_vertex_fns.resize(boost::extents[1+s[0]/2][1+s[1]/2]);

  m_cell_flags.resize((boost::extents[1+s[0]][1+s[1]])); 

  m_cell_pairs.resize((boost::extents[1+s[0]][1+s[1]]));

  for(int y = 0 ; y<=s[1];++y)
    for(int x = 0 ; x<=s[0];++x)
      m_cell_flags[x][y] = CELLFLAG_UNKNOWN;
    
  rect_point_t bl = m_ext_rect.bottom_left();
    
  m_vertex_fns.reindex(bl/2);  
  m_cell_flags.reindex(bl);
  m_cell_pairs.reindex(bl);
}

GridDataset::cellid_t   GridDataset::getCellPairId ( cellid_t c ) const
{
  if(m_cell_flags(c)&CELLFLAG_PAIRED == 0)
    throw std::logic_error("invalid pair requested");

  return m_cell_pairs(c);

}

bool GridDataset::ptLt ( cellid_t c1,cellid_t c2 ) const
{
  double f1 = m_vertex_fns[c1[0]>>1][c1[1]>>1];
  double f2 = m_vertex_fns[c2[0]>>1][c2[1]>>1];

  if( f1 != f2)
    return f1 < f2;

  return c1<c2;

}

GridDataset::cell_fn_t GridDataset::get_cell_fn(cellid_t c) const
{
  if(getCellDim(c) != 0)
    throw std::logic_error("incorrect cell type requsting vert fn value");
  
  c[0] /=2;
  c[1] /=2;

  double ret = m_vertex_fns(c);

  return ret;
}

void GridDataset::set_cell_fn(cellid_t c,cell_fn_t f)
{
  if(getCellDim(c) != 0)
    throw std::logic_error("incorrect cell type requsting vert fn value");

  c[0] /=2;
  c[1] /=2;

  m_vertex_fns(c) = f;
}

uint GridDataset::getCellPoints ( cellid_t c,cellid_t  *p ) const
{
  switch(getCellDim(c))
  {
  case 0:
    p[0] = c;
    return 1;
  case 1:
    {
      cell_coord_t d0 = c[0]&0x01,d1 = c[1]&0x01;
      p[0] = cellid_t(c[0]+d0,c[1]+d1);
      p[1] = cellid_t(c[0]-d0,c[1]-d1);
    }
    return 2;
  case 2:
    p[0] = cellid_t(c[0]+1,c[1]+1);
    p[1] = cellid_t(c[0]+1,c[1]-1);
    p[2] = cellid_t(c[0]-1,c[1]-1);
    p[3] = cellid_t(c[0]-1,c[1]+1);
    return 4;
  default:
    throw std::logic_error("impossible dim");
    return 0;
  }
}

uint GridDataset::getCellFacets ( cellid_t c,cellid_t *f) const
{
  switch(getCellDim(c))
  {
  case 0:
    return 0;
  case 1:
    {
      cell_coord_t d0 = c[0]&0x01,d1 = c[1]&0x01;
      f[0] = cellid_t(c[0]+d0,c[1]+d1);
      f[1] = cellid_t(c[0]-d0,c[1]-d1);
    }
    return 2;
  case 2:
    f[0] = cellid_t(c[0]  ,c[1]+1);
    f[1] = cellid_t(c[0]  ,c[1]-1);
    f[2] = cellid_t(c[0]-1,c[1]  );
    f[3] = cellid_t(c[0]+1,c[1]  );
    return 4;
  default:
    throw std::logic_error("impossible dim");
    return 0;
  }
}

uint GridDataset::getCellCofacets ( cellid_t c,cellid_t *cf) const
{
  cell_coord_t d0 = c[0]&0x01,d1 = c[1]&0x01;

  uint cf_ct = 0;

  switch(getCellDim(c))
  {
  case 0:
    cf[0] = cellid_t(c[0]  ,c[1]+1);
    cf[1] = cellid_t(c[0]  ,c[1]-1);
    cf[2] = cellid_t(c[0]-1,c[1]  );
    cf[3] = cellid_t(c[0]+1,c[1]  );
    cf_ct =  4;
    break;
  case 1:
    {
      cell_coord_t d0 = c[0]&0x01,d1 = c[1]&0x01;
      cf[0] = cellid_t(c[0]+d1,c[1]+d0);
      cf[1] = cellid_t(c[0]-d1,c[1]-d0);
      cf_ct =  2;
    }
    break;
  case 2:
    return 0;
  default:
    throw std::logic_error("impossible dim");
    return 0;
  }

  // position in cf[] where the next valid cf should be placed
  uint cf_nv_pos = 0;

  for( uint i = 0 ;i < cf_ct;++i)
    if(m_ext_rect.contains(cf[i]))
      cf[cf_nv_pos++] = cf[i];

  return cf_nv_pos;

}

uint GridDataset::getMaxCellDim() const
{
  return 2;
}

bool GridDataset::isPairOrientationCorrect ( cellid_t c, cellid_t p) const
{
  return (getCellDim(c)<getCellDim(p));
}

bool GridDataset::isCellMarked ( cellid_t c ) const
{
  return !(m_cell_flags(c) == CELLFLAG_UNKNOWN);
}

bool GridDataset::isCellCritical ( cellid_t c) const
{
  return (m_cell_flags(c) & CELLFLAG_CRITCAL);
}
bool GridDataset::isCellPaired(  cellid_t c) const
{
  return (m_cell_flags(c) & CELLFLAG_PAIRED);
}

void GridDataset::pairCells ( cellid_t c,cellid_t p)
{
  m_cell_pairs(c) = p;
  m_cell_pairs(p) = c;

  m_cell_flags(c) = m_cell_flags(c)|CELLFLAG_PAIRED;
  m_cell_flags(p) = m_cell_flags(p)|CELLFLAG_PAIRED;
}

void GridDataset::markCellCritical ( cellid_t c )
{
  m_cell_flags(c) = m_cell_flags(c)|CELLFLAG_CRITCAL;

  m_critical_cells.push_back(c);
}

bool GridDataset::isTrueBoundryCell ( cellid_t c ) const
{
  return ( m_ext_rect.isOnBoundry(c));
}

bool GridDataset::isFakeBoundryCell ( cellid_t c ) const
{
  return (m_rect.isOnBoundry(c) && (!m_ext_rect.isOnBoundry(c)));
}

bool GridDataset::isCellExterior ( cellid_t c ) const
{
  return (!m_rect.contains(c) && m_ext_rect.contains(c));
}

void GridDataset::connectCps ( cellid_t c1, cellid_t c2)
{
  if(getCellDim(c1) <getCellDim(c2) )
    std::swap(c1,c2);

  if(getCellDim(c1) != getCellDim(c2)+1)
    throw std::logic_error("must connect i,i+1 cp (or vice versa)");

  if(m_msgraph.m_id_cp_map.find(c1) == m_msgraph.m_id_cp_map.end())
    throw std::logic_error(_SSTR("cell not in id_cp_map c1="<<c1));

  if(m_msgraph.m_id_cp_map.find(c2) == m_msgraph.m_id_cp_map.end())
    throw std::logic_error(_SSTR("cell not in id_cp_map c2="<<c2));

  uint cp1_ind = m_msgraph.m_id_cp_map[c1];
  uint cp2_ind = m_msgraph.m_id_cp_map[c2];

  critpt_t *cp1 = m_msgraph.m_cps[cp1_ind];
  critpt_t *cp2 = m_msgraph.m_cps[cp2_ind];

  cp1->des.insert(cp2_ind);
  cp2->asc.insert(cp1_ind);
}

std::string GridDataset::getCellFunctionDescription ( cellid_t c ) const
{
  std::stringstream ss;

  ((std::ostream &)ss)<<c;

  return ss.str();

}

std::string GridDataset::getCellDescription ( cellid_t c ) const
{

  std::stringstream ss;

  ((std::ostream &)ss)<<c;

  return ss.str();

}

void  GridDataset::assignGradients()
{
  // determine all the pairings of all cells in m_rect
  for(cell_coord_t y = m_rect.bottom(); y <= m_rect.top();y += 1)
    for(cell_coord_t x = m_rect.left(); x <= m_rect.right();x += 1)
    {
    cellid_t c(x,y),p;

    if (isCellMarked(c))
      continue;

    if(minPairable_cf(this,c,p))
      pairCells(c,p);
  }

  for(cell_coord_t y = m_rect.bottom(); y <= m_rect.top();y += 1)
    for(cell_coord_t x = m_rect.left(); x <= m_rect.right();x += 1)
    {
    cellid_t c(x,y);

    if (!isCellMarked(c)) markCellCritical(c);
  }

  // mark artificial boundry as critical

  for(cell_coord_t x = m_rect.left(); x <= m_rect.right();x += 1)
  {
    cellid_t bcs[] = {cellid_t(x,m_rect.bottom()),cellid_t(x,m_rect.top())};

    for(uint i = 0 ; i <sizeof(bcs)/sizeof(cellid_t);++i)
    {
      cellid_t &c = bcs[i];      
      
      if(c == cellid_t(36,40))
        _LOG(c<<getCellPairId(c)<<m_rect<<m_ext_rect);
      
      if(isCellCritical(c)) continue;
      
      cellid_t cf[20];
      
      u_int cf_ct =  getCellCofacets(c,cf);      
      
      for(u_int j = 0 ; j <cf_ct;++j)
      {
        if(isCellExterior(cf[j]))
        {
          markCellCritical(c);
          markCellCritical(getCellPairId(c));
        }
      }      
    }
  }

  for(cell_coord_t y = m_rect.bottom()+1; y < m_rect.top();y += 1)
  {
    cellid_t bcs[] = {cellid_t(m_rect.left(),y),cellid_t(m_rect.right(),y)};

    for(uint i = 0 ; i <sizeof(bcs)/sizeof(cellid_t);++i)
    {
      cellid_t &c = bcs[i];
      
      if(isCellCritical(c)) continue;
      
      cellid_t cf[20];
      
      u_int cf_ct =  getCellCofacets(c,cf);      
      
      for(u_int j = 0 ; j <cf_ct;++j)
      {
        if(isCellExterior(cf[j]))
        {
          markCellCritical(c);
          markCellCritical(getCellPairId(c));
        }
      }
    }
  }
}

inline void add_to_grad_tree_proxy(GridDataset::cellid_t )
{
  // just do nothing
  return;
}

void  GridDataset::computeDiscs()
{

  addCriticalPointsToMSComplex
      (&m_msgraph,this,m_critical_cells.begin(),m_critical_cells.end());     

  for ( cellid_list_t::iterator it = m_critical_cells.begin() ;
  it != m_critical_cells.end();++it)
  {

    switch(getCellDim(*it))
    {
    case 0:
      track_gradient_tree_bfs
          (this,*it,GRADIENT_DIR_UPWARD,
           add_to_grad_tree_proxy,
           boost::bind(&GridDataset::connectCps,this,*it,_1));
      break;
    case 2:
      track_gradient_tree_bfs
          (this,*it,GRADIENT_DIR_DOWNWARD,
           add_to_grad_tree_proxy,
           boost::bind(&GridDataset::connectCps,this,_1,*it));
      break;
    default:
      break;
    }
  }
  
  m_critical_cells.clear();

  // mark all the boundry cps as boundry cancellable

  for(cell_coord_t x = m_rect.left(); x <= m_rect.right();x += 1)
  {
    cellid_t bcs[] = {cellid_t(x,m_rect.bottom()),cellid_t(x,m_rect.top())};

    for(uint i = 0 ; i <sizeof(bcs)/sizeof(cellid_t);++i)
    {
      if(m_msgraph.m_id_cp_map.find(bcs[i]) == m_msgraph.m_id_cp_map.end())
        continue;
      
      uint cp_idx = m_msgraph.m_id_cp_map[bcs[i]];
      m_msgraph.m_cps[cp_idx]->isBoundryCancelable = true;
    }
  }

  for(cell_coord_t y = m_rect.bottom()+1; y < m_rect.top();y += 1)
  {
    cellid_t bcs[] = {cellid_t(m_rect.left(),y),cellid_t(m_rect.right(),y)};

    for(uint i = 0 ; i <sizeof(bcs)/sizeof(cellid_t);++i)
    {
      uint cp_idx = m_msgraph.m_id_cp_map[bcs[i]];
      m_msgraph.m_cps[cp_idx]->isBoundryCancelable = true;
    }
  }
}

void GridDataset::getCellCoord(cellid_t c,double &x,double &y,double &z)
{
  x = c[0];  y = 0;  z = c[1];

  cellid_t pts[20];

  uint pts_ct = getCellPoints(c,pts);

  for( int j = 0 ; j <pts_ct ;++j)
    y += get_cell_fn(pts[j]);

  y /= pts_ct;
}
