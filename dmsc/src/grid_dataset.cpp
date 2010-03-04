#include <grid_dataset.h>

#include <vector>

typedef GridDataset::cellid_t cellid_t;

GridDataset::GridDataset(const rect_t &r,const rect_t &e):
    m_rect(r),m_ext_rect(e)
{

  // TODO: assert that the given rect is of even size..
  //       since each vertex is in the even positions
  //

  rect_size_t s = m_ext_rect.size();

  m_vertex_fns.resize(boost::extents[1+s[0]/2][1+s[1]/2]);

  for(int i = 0 ; i< 1+s[0];++i)
    std::fill(m_cell_flags[i].begin(),m_cell_flags[i].end(),CELLFLAG_UNKNOWN);


}

void GridDataset::set_datarow(const double *data, uint rownum)
{
  std::copy(data,data+m_ext_rect.size()[0]+1,m_vertex_fns[rownum].begin());
}

GridDataset::cellid_t   GridDataset::getCellPairId ( cellid_t c ) const
{
  if(m_cell_flags(c)&CELLFLAG_PAIRED == 0)
    throw std::logic_error("invalid pair requested");

  return m_cell_pairs(c);

}

bool GridDataset::ptLt ( cellid_t c1,cellid_t c2 ) const
{
  if(!isPoint(c1) || !isPoint(c2))
    throw std::logic_error("this is used only to compare points");

  std::for_each(c1.begin(),c1.end(),
                boost::bind(shift_right<cell_coord_t,uint>,_1,1));

  std::for_each(c2.begin(),c2.end(),
                boost::bind(shift_right<cell_coord_t,uint>,_1,1));

  if(m_vertex_fns(c1) != m_vertex_fns(c2))
    return m_vertex_fns(c1) < m_vertex_fns(c2);

  return c1<c2;

}

bool GridDataset::isPoint(cellid_t c)
{
  if(c[0]&0x01 == 1 ||c[1]&0x01 == 1 )
    return false;

  return true;
}

uint GridDataset::getCellPoints ( cellid_t c,cellid_t  *p ) const
{
  cellid_t d(c[0]&0x01,c[1]&0x01);

  switch(getCellDim(c))
  {
  case 0:
    p[0] = c;
    return 1;
  case 1:
    p[0] = cellid_t(c[0]+d[0],c[1]+d[1]);
    p[1] = cellid_t(c[0]-d[0],c[1]-d[1]);
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
  cellid_t d(c[0]&0x01,c[1]&0x01);

  switch(getCellDim(c))
  {
  case 0:
    throw std::logic_error("zero dim cell has no facets");
    return 0;
  case 1:
    f[0] = cellid_t(c[0]+d[0],c[1]+d[1]);
    f[1] = cellid_t(c[0]-d[0],c[1]-d[1]);
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
  cellid_t d(c[0]&0x01,c[1]&0x01);

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
    cf[0] = cellid_t(c[0]+d[1],c[1]+d[0]);
    cf[1] = cellid_t(c[0]-d[1],c[1]-d[0]);
    cf_ct =  2;
    break;
  case 2:
    throw std::logic_error("2 dim cell has no co-facets");
    return 0;
  default:
    throw std::logic_error("impossible dim");
    return 0;
  }

  uint cf_nv_pos = 0; // position in cf[] where the next valid cf
                      // should be placed

  for( uint i = 0 ;i < cf_ct;++i)
    if(m_ext_rect.contains(cf[i]))
      cf[cf_nv_pos++] = cf[i];

  return cf_nv_pos;

}

uint GridDataset::getCellDim ( cellid_t c ) const
{
  return (c[0]&0x01 + c[1]&0x01);
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
  return (m_cell_flags(c) == CELLFLAG_CRITCAL);
}

void GridDataset::pairCells ( cellid_t c,cellid_t p)
{
  m_cell_pairs(c) = p;
  m_cell_pairs(p) = c;

  m_cell_flags(c) = (eCellFlags)((uint) m_cell_flags(c)|CELLFLAG_PAIRED);
  m_cell_flags(p) = (eCellFlags)((uint) m_cell_flags(p)|CELLFLAG_PAIRED);
}

void GridDataset::markCellCritical ( cellid_t c )
{
  m_cell_flags(c) = (eCellFlags)((uint) m_cell_flags(c)|CELLFLAG_CRITCAL);
}

bool GridDataset::isTrueBoundryCell ( cellid_t c ) const
{
  return (m_rect.isOnBoundry(c) && m_ext_rect.isOnBoundry(c));
}

bool GridDataset::isFakeBoundryCell ( cellid_t c ) const
{
  return (m_rect.isOnBoundry(c) && (!m_ext_rect.isOnBoundry(c)));
}

bool GridDataset::isCellExterior ( cellid_t c ) const
{
  return (!m_rect.contains(c) && m_ext_rect.contains(c));
}

