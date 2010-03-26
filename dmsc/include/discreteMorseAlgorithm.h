/***************************************************************************
 *   Copyright (C) 2009 by nithin,,,   *
 *   nithin@gauss   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef __DISCRETEMORSEALGORITHM_H_INCLUDED
#define __DISCRETEMORSEALGORITHM_H_INCLUDED

#include <queue>
#include <vector>
#include <list>
#include <algorithm>
#include <stack>
#include <cstring>

#include <boost/bind.hpp>

#include <logutil.h>
#include <cpputils.h>

#include <discreteMorseDS.h>

#define LOG_LEVEL 1

#define DEBUG_BEGIN if(LOG_LEVEL > 0) {
#define DEBUG_END }
#define DEBUG_STMT(x) DEBUG_BEGIN x; DEBUG_END;
#define DEBUG_LOG(x) DEBUG_STMT(_LOG(x))

#define DEBUG_BEGIN_V if(LOG_LEVEL > 1) {
#define DEBUG_END_V }
#define DEBUG_STMT_V(x) DEBUG_BEGIN_V x; DEBUG_END_V;
#define DEBUG_LOG_V(x) DEBUG_STMT_V(_LOG(x))

#define DEBUG_BEGIN_VV if(LOG_LEVEL > 2) {
#define DEBUG_END_VV }
#define DEBUG_STMT_VV(x) DEBUG_BEGIN_VV x; DEBUG_END_VV;
#define DEBUG_LOG_VV(x) DEBUG_STMT_VV(_LOG(x))

template <typename id_type>
    bool compareCells_gen ( const IDiscreteDataset<id_type> *dataset,id_type cellid1,id_type cellid2 )
{
  id_type id_typearr[20];

  id_type *cell1pts = id_typearr + 0;
  id_type *cell2pts = id_typearr + 10;

  uint cell1pts_ct = dataset->getCellPoints ( cellid1,cell1pts );
  uint cell2pts_ct = dataset->getCellPoints ( cellid2,cell2pts );

  std::sort ( cell1pts,cell1pts+cell1pts_ct,boost::bind ( &IDiscreteDataset<id_type>::ptLt,dataset,_2,_1 ) );
  std::sort ( cell2pts,cell2pts+cell2pts_ct,boost::bind ( &IDiscreteDataset<id_type>::ptLt,dataset,_2,_1 ) );

  return std::lexicographical_compare ( cell1pts,cell1pts+cell1pts_ct,cell2pts,cell2pts+cell2pts_ct,boost::bind ( &IDiscreteDataset<id_type>::ptLt,dataset,_1,_2 ) );
};

template <typename id_type>

    bool minPairable_cf
    (
        IDiscreteDataset<id_type> *dataset,
        id_type cellId,
        id_type& pairid
        )
{
  id_type cofacets[20];
  bool    cofacet_usable[20];

  uint cofacet_count = dataset->getCellCofacets ( cellId,cofacets );

  bool isTrueBoundryCell = dataset->isTrueBoundryCell ( cellId ) ;

  // for each co facet
  for ( uint i = 0 ; i < cofacet_count ; i++ )
  {
    id_type facets[20];
    uint facet_count = dataset->getCellFacets ( cofacets[i],facets );

    cofacet_usable[i] = true;

    if ( isTrueBoundryCell &&
         !dataset->isTrueBoundryCell ( cofacets[i] ) )
    {
      cofacet_usable[i] = false;
      continue;
    }

    for ( uint j = 0 ; j < facet_count ; j++ )
    {
      if ( dataset->compareCells ( cellId,facets[j] ) )
      {
        cofacet_usable[i] = false;
        break;
      }
    }
  }

  bool pairid_usable = false;

  for ( uint i =0 ; i < cofacet_count;i++ )
  {
    if ( cofacet_usable[i] == false )
      continue;

    if(pairid_usable == false)
    {
      pairid_usable = true;
      pairid = cofacets[i];
      continue;
    }

    if ( dataset->compareCells ( cofacets[i],pairid ) )
      pairid = cofacets[i];

  }
  return pairid_usable;
}



template <typename id_type,typename cell_iter_type,typename add_to_cancellable_list_ftor_t >
    void assignGradient
    (
        IDiscreteDataset<id_type> *dataset,
        cell_iter_type start,
        cell_iter_type end,
        add_to_cancellable_list_ftor_t add_to_cancellable_list_ftor
        )
{
  for ( cell_iter_type iter = start;iter != end;++iter )
  {
    if ( dataset->isCellMarked ( *iter ) ) continue;

    id_type pairid;

    if ( minPairable_cf ( dataset,*iter ,pairid ) == true )
    {
      if ( dataset->isFakeBoundryCell ( *iter) &&
           (!dataset->isFakeBoundryCell ( pairid)))
      {
        add_to_cancellable_list_ftor ( *iter,pairid );
        dataset->markCellCritical ( *iter );
      }
      else
        dataset->pairCells ( pairid,*iter );
    }
    else
    {
      dataset->markCellCritical ( *iter );
    }
  }
}

template <typename id_type,typename cell_iter_type,typename add_to_cancellable_list_ftor_t >
    void assignGradient
    (
        IDiscreteDataset<id_type> *dataset,
        std::pair<cell_iter_type,cell_iter_type> iters,
        add_to_cancellable_list_ftor_t add_to_cancellable_list_ftor
        )
{
  assignGradient ( dataset,iters.first,iters.second,add_to_cancellable_list_ftor );
}

template
    <
    typename id_type,
    typename add_to_grad_tree_functor_type,
    typename add_to_incident_crit_pts_functor_type
    >
    void track_gradient_tree_bfs
    (
        IDiscreteDataset<id_type> *dataset,
        id_type start_cellId,
        eGradDirection gradient_dir,
        add_to_grad_tree_functor_type add_to_grad_tree_functor,
        add_to_incident_crit_pts_functor_type add_to_incident_crit_pts_functor
        )
{
  static uint ( IDiscreteDataset<id_type>::*getcets[2] ) ( id_type,id_type * ) const =
  {
    &IDiscreteDataset<id_type>::getCellFacets,
    &IDiscreteDataset<id_type>::getCellCofacets
  };

  std::queue<id_type> cell_queue;

  // mark here that that cellid has no parent.

  cell_queue.push ( start_cellId );

  while ( !cell_queue.empty() )
  {
    id_type top_cell = cell_queue.front();

    cell_queue.pop();

    add_to_grad_tree_functor ( top_cell );

    id_type      cets[20];

    uint cet_ct = ( dataset->*getcets[gradient_dir] ) ( top_cell,cets );

    for ( uint i = 0 ; i < cet_ct ; i++ )
    {
      if ( dataset->isCellCritical ( cets[i] ) )
      {
        add_to_incident_crit_pts_functor ( cets[i] );
      }
      else
      {
        if ( !dataset->isCellExterior ( cets[i] ) )
        {
          id_type next_cell = dataset->getCellPairId ( cets[i] );

          if ( dataset->getCellDim ( top_cell ) ==
               dataset->getCellDim ( next_cell ) &&
               next_cell != top_cell )
          {
            // mark here that the parent of next cell is top_cell
            cell_queue.push ( next_cell );
          }
        }
      }
    }
  }
}

template <typename id_t,typename cpiter_t>
    void addCriticalPointsToMSComplex
    (
        MSComplex<id_t> *msc,
        cpiter_t begin,
        cpiter_t end
        )
{
  typedef typename MSComplex<id_t>::critical_point cp_t;
  typedef typename std::map<id_t,uint>              id_cp_map_t;

  msc->m_cps.resize(end-begin);

  uint pos = 0;

  for ( cpiter_t it = begin;it != end ; ++it )
  {
    cp_t * cp                 = new cp_t;
    msc->m_cps[pos]           = cp;
    cp->cellid                = *it;
    msc->m_id_cp_map.insert ( std::make_pair ( *it,pos ) );

    ++pos;
  }
}

template <typename id_t,typename add_id_to_set_ftor_t>
    void add_to_grad_tree_func
    (
        IDiscreteDataset<id_t> *ds,
        eGradDirection dir,
        id_t cellid,
        add_id_to_set_ftor_t add_id_to_set_ftor
        )
{
  static uint ( IDiscreteDataset<id_t>::*getcets[2] ) ( id_t,id_t * ) const =
  {
    &IDiscreteDataset<id_t>::getCellFacets,
    &IDiscreteDataset<id_t>::getCellCofacets
  };

  std::queue<id_t> cets_queue;

  if ( add_id_to_set_ftor ( cellid ) )

    cets_queue.push ( cellid );

  while ( !cets_queue.empty() )
  {
    cellid = cets_queue.front();

    cets_queue.pop();

    id_t cets[20];

    uint cet_ct = ( ds->*getcets[dir] ) ( cellid,cets );

    for ( uint i = 0 ; i <cet_ct;i++ )
    {
      if ( add_id_to_set_ftor ( cets[i] ) )
      {
        cets_queue.push ( cets[i] );
      }
    }
  }
}

template
    <
    typename id_type,
    typename add_id_to_set_ftor_t,
    typename add_to_incident_crit_pts_functor_type
    >
    void track_gradient_tree_with_closure
    (
        IDiscreteDataset<id_type> *dataset,
        id_type start_cellId,
        eGradDirection gradient_dir,
        add_id_to_set_ftor_t add_id_to_set_ftor,
        add_to_incident_crit_pts_functor_type add_to_incident_crit_pts_functor
        )
{
  track_gradient_tree_bfs
      (
          dataset,
          start_cellId,
          gradient_dir,
          boost::bind ( add_to_grad_tree_func<id_t,add_id_to_set_ftor_t>,dataset,gradient_dir,_1,add_id_to_set_ftor ),
          add_to_incident_crit_pts_functor
          );

}


template<typename id1_t,typename id2_t,typename converter_ftor_t>
    void mergeDiscs ( std::set<id1_t> *disc1,const std::set<id2_t> *disc2,converter_ftor_t converter_ftor )
{
  for ( typename std::set<id2_t>::iterator cell_it = disc2->begin();cell_it != disc2->end(); ++cell_it )
    disc1->insert ( converter_ftor ( *cell_it ) );
}

#undef LOG_LEVEL
#define LOG_LEVEL 1

template <typename id_t>
    void cancelPairs ( MSComplex<id_t> *msc,uint cp0_ind,uint cp1_ind )
{

  typedef typename std::multiset<uint> cp_adj_t;

  DEBUG_LOG_V ( "=========================" );
  DEBUG_LOG_V ( "cancelling pairs "<<msc->m_cps[cp0_ind]->cellid<<" "<<msc->m_cps[cp1_ind]->cellid );
  DEBUG_LOG_V ( "-------------------------" );

  if ( msc->m_cps[cp0_ind]->asc.find ( cp1_ind ) == msc->m_cps[cp0_ind]->asc.end() )
  {
    std::swap ( cp0_ind,cp1_ind );
  }


  if ( msc->m_cps[cp0_ind]->asc.find ( cp1_ind ) == msc->m_cps[cp0_ind]->asc.end() )
  {
    _ERROR ( "pair =  "<<msc->m_cps[cp0_ind]->cellid<<" "<<msc->m_cps[cp1_ind]->cellid );
    _ERROR ( "critical points are not connected" );
    return;
  }



  //   std::stringstream ss;
  //
  //   ss<<"asc("<<msc->m_cps[cp0_ind]->cellid<<")="<<"{";
  //
  //   for ( typename cp_adj_t::iterator asc0_it = msc->m_cps[cp0_ind]->asc.begin();asc0_it != msc->m_cps[cp0_ind]->asc.end(); ++asc0_it )
  //   {
  //     ss<<msc->m_cps[*asc0_it]->cellid<<",";
  //   }
  //   ss<<"}";
  //
  //   _LOG ( ss.str() );


  for ( typename cp_adj_t::iterator asc0_it = msc->m_cps[cp0_ind]->asc.begin();asc0_it != msc->m_cps[cp0_ind]->asc.end(); ++asc0_it )
  {
    for ( typename cp_adj_t::iterator des1_it = msc->m_cps[cp1_ind]->des.begin();des1_it != msc->m_cps[cp1_ind]->des.end(); ++des1_it )
    {
      if ( ( *asc0_it == cp1_ind ) || ( *des1_it == cp0_ind ) )
        continue;

      msc->m_cps[ *asc0_it ]->des.insert ( *des1_it );
      msc->m_cps[ *des1_it ]->asc.insert ( *asc0_it );

      DEBUG_LOG_V ( "Connected "<<msc->m_cps[*asc0_it]->cellid<<" to "<<msc->m_cps[*des1_it]->cellid );

      if ( msc->m_cps[ *asc0_it ]->des.count ( *des1_it ) >= 2 )
      {
        msc->m_cps[ *asc0_it ]->isOnStrangulationPath = true;
        msc->m_cps[ *des1_it ]->isOnStrangulationPath = true;

        DEBUG_LOG_V ( "Strangulation by "<<msc->m_cps[*asc0_it]->cellid<<" "<<msc->m_cps[*des1_it]->cellid );

      }
    }
  }

  for ( typename cp_adj_t::iterator asc0_it = msc->m_cps[cp0_ind]->asc.begin();asc0_it != msc->m_cps[cp0_ind]->asc.end(); ++asc0_it )
  {
    if ( *asc0_it == cp1_ind )  continue;

//    DEBUG_LOG_V ( "merging des_manifold of "<< cp1_ind<<" with " << *asc0_it );

//    msc->m_cps[ *asc0_it ]->des_disc.insert ( msc->m_cps[ cp1_ind ]->des_disc.begin(),msc->m_cps[ cp1_ind ]->des_disc.end() );

    DEBUG_LOG_V ( "Removing   "<<cp0_ind<<" from "<< ( *asc0_it ) );

    msc->m_cps[  *asc0_it ]->des.erase ( cp0_ind );
  }

  for ( typename cp_adj_t::iterator des0_it = msc->m_cps[cp0_ind]->des.begin();des0_it != msc->m_cps[cp0_ind]->des.end(); ++des0_it )
  {
    msc->m_cps[ ( *des0_it ) ]->asc.erase ( cp0_ind );

    DEBUG_LOG_V ( "Removing   "<<cp0_ind<<" from "<< ( *des0_it ) );
  }

  for ( typename cp_adj_t::iterator asc1_it = msc->m_cps[cp1_ind]->asc.begin();asc1_it != msc->m_cps[cp1_ind]->asc.end(); ++asc1_it )
  {
    msc->m_cps[ ( *asc1_it ) ]->des.erase ( cp1_ind );

    DEBUG_LOG_V ( "Removing   "<<cp1_ind<<" from "<< ( *asc1_it ) );
  }

  for ( typename cp_adj_t::iterator des1_it = msc->m_cps[cp1_ind]->des.begin();des1_it != msc->m_cps[cp1_ind]->des.end(); ++des1_it )
  {
    if ( ( *des1_it ) == cp0_ind )  continue;

//    DEBUG_LOG_V ( "merging asc_manifold of "<< cp0_ind<<" with " << *des1_it );

//    msc->m_cps[ *des1_it ]->asc_disc.insert ( msc->m_cps[ cp0_ind ]->asc_disc.begin(),msc->m_cps[ cp0_ind ]->asc_disc.end() );

    DEBUG_LOG_V ( "Removing   "<<cp1_ind<<" from "<< ( *des1_it ) );

    msc->m_cps[ ( *des1_it ) ]->asc.erase ( cp1_ind );


  }

  msc->m_cps[cp0_ind]->isCancelled = true;
  //  msc->m_cps[cp0_ind]->asc.clear();
  msc->m_cps[cp0_ind]->des.clear();
  //  msc->m_cps[cp0_ind]->asc_disc.clear();
  //  msc->m_cps[cp0_ind]->des_disc.clear();

  msc->m_cps[cp1_ind]->isCancelled = true;
  msc->m_cps[cp1_ind]->asc.clear();
  //  msc->m_cps[cp1_ind]->des.clear();
  //  msc->m_cps[cp1_ind]->asc_disc.clear();
  //  msc->m_cps[cp1_ind]->des_disc.clear();

  DEBUG_LOG_V ( "=========================" );
}

#undef LOG_LEVEL
#define LOG_LEVEL 2

template <typename id_t>
    struct simplifyComplex_ftor
{
  typedef typename MSComplex<id_t>::critical_point  cp_t;
  typedef typename std::multiset<uint> cpadj_t;

  static id_t getCpCellid ( MSComplex<id_t> *msc,uint cpind )
  {
    return msc->m_cps[cpind]->cellid;
  }

  void operator() ( MSComplex<id_t> *msc,IDiscreteDataset<id_t> *ds )
  {
    using boost::bind;


    uint num_vert = msc->m_cps.size() ;

    uint *sorted_verts = new uint[num_vert];

    for ( uint i = 0 ; i < num_vert;i++ )
      sorted_verts[i] = i;

    std::sort ( sorted_verts,sorted_verts+num_vert,bind ( compareCells_gen<id_t>,ds,bind ( getCpCellid,msc,_1 ),bind ( getCpCellid,msc,_2 ) ) );

    uint *vert_pos = new uint[num_vert];

    for ( uint i = 0 ; i < num_vert;i++ )
      vert_pos[sorted_verts[i]] = i ;

    DEBUG_STMT_V ( log_range ( vert_pos,vert_pos+num_vert,"Crit pt pos in sorted list" ) );

    DEBUG_STMT_V ( log_range ( sorted_verts,sorted_verts+num_vert,
                               bind ( &IDiscreteDataset<id_t>::getCellFunctionDescription,ds,bind ( getCpCellid,msc,_1 ) ),
                               "vert function description" ) );


    std::vector<std::pair<uint,uint> > msgraph_edges;

    for ( uint i = 0 ; i< num_vert;i++ )
    {
      cp_t *cp = msc->m_cps[i];
      for ( typename cpadj_t::iterator cpadj_it = cp->des.begin();cpadj_it != cp->des.end();++cpadj_it )
      {
        msgraph_edges.push_back ( std::make_pair ( i,*cpadj_it ) );
      }
    }

    uint num_edge = msgraph_edges.size() ;

    uint *edge_vdiff = new uint[num_edge];

    for ( uint i = 0 ; i < num_edge;i++ )
    {
      uint v1 = msgraph_edges[i].first;
      uint v2 = msgraph_edges[i].second;

      edge_vdiff[i] = ( vert_pos[v1]>vert_pos[v2] ) ? ( vert_pos[v1]-vert_pos[v2] ) : ( vert_pos[v2]-vert_pos[v1] );
    }

    DEBUG_STMT_V ( log_range ( edge_vdiff,edge_vdiff+num_edge,"edge vert position diff" ) );

    uint *sorted_edges = new uint[num_edge];

    for ( uint i = 0 ; i < num_edge;i++ )
      sorted_edges[i] = i;

    std::sort ( sorted_edges,sorted_edges+num_edge,bind ( compareListItems<uint,uint>,edge_vdiff,_1,_2 ) );

    DEBUG_STMT_V ( log_range ( sorted_edges,sorted_edges+num_edge,"edges in sorted order  " ) );

    uint num_cancellations = 0 ;
    uint max_cancellations = 0;

    for ( uint i = 0 ; i < num_edge;i++ )
    {
      uint edge_ind = sorted_edges[i];

      uint v1 = msgraph_edges[edge_ind].first;
      uint v2 = msgraph_edges[edge_ind].second;

      if (
          ( msc->m_cps[v1]->isCancelled           == false ) &&
          ( msc->m_cps[v2]->isCancelled           == false ) &&
          ( msc->m_cps[v1]->isOnStrangulationPath == false ) &&
          ( msc->m_cps[v2]->isOnStrangulationPath == false ) &&
          ( ! ds->isFakeBoundryCell ( msc->m_cps[v1]->cellid ) ) &&
          ( ! ds->isFakeBoundryCell ( msc->m_cps[v2]->cellid ) )

          )
      {
        if ( num_cancellations >= max_cancellations ) break;
        cancelPairs ( msc,v1,v2 );
        num_cancellations++;
      }
    }

    DEBUG_LOG_V ( "No of cancellations ::" <<num_cancellations );

    delete []sorted_verts;
    delete []vert_pos;
    delete []edge_vdiff;
    delete []sorted_edges;
  }
};

template <typename id_t>
    void simplifyComplex ( MSComplex<id_t> *msc,IDiscreteDataset<id_t> *ds )
{
  simplifyComplex_ftor<id_t>() ( msc,ds );
}

template <typename id_t,typename unicellid_t,typename converter_ftor_t >
    void convertMSComplexToGeneric ( const MSComplex<id_t> *ms_in,MSComplex<unicellid_t> *&ms_out,converter_ftor_t converter_ftor )
{
  ms_out->m_cps.resize(ms_in->m_cps.size());

  for ( uint i = 0 ; i< ms_in->m_cps.size();i++ )
  {
    ms_out->m_cps[i] = new typename MSComplex<unicellid_t >::critical_point;

    ms_out->m_cps[i]->cellid = converter_ftor ( ms_in->m_cps[i]->cellid ) ;

    ms_out->m_cps[i]->isOnStrangulationPath = ms_in->m_cps[i]->isOnStrangulationPath;

    ms_out->m_cps[i]->isCancelled = ms_in->m_cps[i]->isCancelled;

    ms_out->m_cps[i]->isBoundryCancelable = ms_in->m_cps[i]->isBoundryCancelable;

    ms_out->m_cps[i]->asc.insert ( ms_in->m_cps[i]->asc.begin(),ms_in->m_cps[i]->asc.end() );

    ms_out->m_cps[i]->des.insert ( ms_in->m_cps[i]->des.begin(),ms_in->m_cps[i]->des.end() );

    mergeDiscs ( &ms_out->m_cps[i]->asc_disc,&ms_in->m_cps[i]->asc_disc,converter_ftor );

    mergeDiscs ( &ms_out->m_cps[i]->des_disc,&ms_in->m_cps[i]->des_disc,converter_ftor );

    ms_out->m_id_cp_map.insert ( std::make_pair ( converter_ftor ( ms_in->m_cps[i]->cellid ),i ) );

  }
}


template <typename id_t>
    void union_complexes_up
    ( MSComplex<id_t>  *msc1,
      MSComplex<id_t>  *msc2 ,
      std::vector<std::pair<id_t,id_t> > &cancellable_pairs)
{

  typedef typename MSComplex<id_t>::critical_point critical_point_t;
  typedef typename MSComplex<id_t>::critical_point::connection_t connection_t;
  typedef typename MSComplex<id_t>::id_cp_map_t id_cp_map_t;
  typedef typename MSComplex<id_t>::cp_ptr_list_t cp_ptr_list_t;
  typedef typename std::pair<id_t,id_t> cancel_pair_t;
  typedef typename std::vector<cancel_pair_t> cancel_pair_list_t;

  // create a union of the 2 complexes in msc1

  std::set<uint> common_boundry_cps;

  for(uint i = 0 ; i <msc2->m_cps.size();++i)
  {
    critical_point_t *src_cp = msc2->m_cps[i];

    if(src_cp->isCancelled)
      continue;

    typename id_cp_map_t::iterator dest_it
        = msc1->m_id_cp_map.find(src_cp->cellid);

    // it is not present in msc1
    if( dest_it == msc1->m_id_cp_map.end())
    {
      // all uncommon bc pairs must be cancelled earlier.
      if(src_cp->isBoundryCancelable == true)
        throw std::logic_error("missing common bc in msc1");

      // allocate a new cp
      critical_point_t* dest_cp = new critical_point_t;

      // copy over the trivial data
      dest_cp->isCancelled               = src_cp->isCancelled;
      dest_cp->isBoundryCancelable       = src_cp->isBoundryCancelable;
      dest_cp->isOnStrangulationPath     = src_cp->isOnStrangulationPath;
      dest_cp->cellid                    = src_cp->cellid;

      msc1->m_id_cp_map[dest_cp->cellid] = msc1->m_cps.size();
      msc1->m_cps.push_back(dest_cp);

    }
    else
    {
      common_boundry_cps.insert(dest_it->second);
    }
  }

  // copy over the ascending descending connections..
  // careful not duplicate boundry connections.

  for(uint i = 0 ; i <msc2->m_cps.size();++i)
  {
    critical_point_t *src_cp = msc2->m_cps[i];

    if(src_cp->isCancelled)
      continue;

    if(msc1->m_id_cp_map.find(src_cp->cellid) ==
       msc1->m_id_cp_map.end())
      throw std::logic_error("could not find cp in msc1");

    uint dest_idx = msc1->m_id_cp_map[src_cp->cellid];

    critical_point_t *dest_cp = msc1->m_cps[dest_idx];

    bool is_common_boundry_cp =
        (common_boundry_cps.find(dest_idx) !=common_boundry_cps.end());

    connection_t * src_asc_des[] = {&src_cp->asc,&src_cp->des};
    connection_t * dest_asc_des[] = {&dest_cp->asc,&dest_cp->des};

    for(uint j = 0 ; j < 2; ++j)
    {
      for(typename connection_t::iterator conn_it = src_asc_des[j]->begin();
      conn_it != src_asc_des[j]->end(); ++conn_it)
      {

        if(*conn_it >= msc2->m_cps.size())
        {
          std::stringstream ss;

          ss<<"invalid idx in msc2"<<std::endl;
          ss<<"*conn_it"<<*conn_it;
          ss<<"msc2->m_cps.size()"<<msc2->m_cps.size();

          throw std::logic_error(ss.str());
        }

        id_t conn_cp_id = msc2->m_cps[*conn_it]->cellid;

        if(msc1->m_id_cp_map.find(conn_cp_id) ==
           msc1->m_id_cp_map.end())
        {
          std::stringstream ss;

          ss<<"missing cp in msc1 "<<std::endl;
          ss<<"dest cp_id "<<dest_cp->cellid<<std::endl;
          ss<<"conn_cp_id "<<conn_cp_id<<std::endl;
          ss<<"is_cancelled "<<msc2->m_cps[*conn_it]->isCancelled<<std::endl;

          throw std::logic_error(ss.str());
        }

        uint conn_cp_idx = msc1->m_id_cp_map[conn_cp_id];

        // if a common boundry cp is connected to another common boundry cp
        // then this is no new information because the connection path must
        // be restricted to the boundry.

        if(is_common_boundry_cp &&
           (common_boundry_cps.find(conn_cp_idx) !=common_boundry_cps.end()))
          continue;

        dest_asc_des[j]->insert(conn_cp_idx);
      }
    }
  }

  cancel_pair_list_t cancelled_pairs;

  // carry out cancellation
  for(typename cancel_pair_list_t::const_iterator it = cancellable_pairs.begin();
  it != cancellable_pairs.end() ; ++it)
  {

    id_t cp1_id = it->first;
    id_t cp2_id = it->second;

    if(msc1->m_id_cp_map.find(cp1_id) ==
       msc1->m_id_cp_map.end())
      continue;

    if(msc1->m_id_cp_map.find(cp2_id) ==
       msc1->m_id_cp_map.end())
      continue;

    uint cp1_idx = msc1->m_id_cp_map[cp1_id];
    uint cp2_idx = msc1->m_id_cp_map[cp2_id];

    if(common_boundry_cps.find(cp1_idx) ==
       common_boundry_cps.end())
      continue;

    if(common_boundry_cps.find(cp2_idx) ==
       common_boundry_cps.end())
      continue;

    cancelPairs(msc1,cp1_idx,cp2_idx);

    cancelled_pairs.push_back(*it);
  }

  // in the cancelpairs list maintain only those that were infact cancelled
  std::copy(cancelled_pairs.begin(),cancelled_pairs.end(),
            cancellable_pairs.begin());

  cancellable_pairs.resize(cancelled_pairs.size());
}

template <typename id_t>
    void uncancel_pairs
    ( MSComplex<id_t>  *msc,
      uint cp1_idx,
      uint cp2_idx,
      const std::map<id_t,id_t> &cancel_pair_map)
{

  typedef typename MSComplex<id_t>::critical_point critical_point_t;
  typedef typename MSComplex<id_t>::critical_point::connection_t connection_t;
  typedef typename connection_t::iterator connection_iter_t;
  typedef typename MSComplex<id_t>::id_cp_map_t id_cp_map_t;
  typedef typename MSComplex<id_t>::cp_ptr_list_t cp_ptr_list_t;
  typedef typename std::pair<id_t,id_t> cancel_pair_t;
  typedef typename std::vector<cancel_pair_t> cancel_pair_list_t;
  typedef typename std::map<id_t,id_t> cancel_pair_map_t;
  typedef typename cancel_pair_map_t::const_iterator cancel_pair_map_citer_t;

  if( msc->m_cps[cp1_idx]->asc.find ( cp2_idx ) ==
      msc->m_cps[cp1_idx]->asc.end() )
  {
    std::swap ( cp1_idx,cp2_idx );
  }

  connection_t * acdc_conns[] =
  {&msc->m_cps[cp1_idx]->asc,
   &msc->m_cps[cp2_idx]->des};

  uint * cp_idxs[]= {&cp1_idx,&cp2_idx};


  for(uint i = 0 ;i < 2 ; ++i)
  {

    if( acdc_conns[i]->find ( *cp_idxs[(i+1)%2] ) ==
        acdc_conns[i]->end())
    {
      throw std::logic_error("cancellable pair is not connected");
    }

    connection_t new_acdc;

    for(connection_iter_t acdc_conn_it = acdc_conns[i]->begin();
        acdc_conn_it != acdc_conns[i]->end();++acdc_conn_it)
    {
      if(*acdc_conn_it == *cp_idxs[(i+1)%2])
        continue;

      if(msc->m_cps[*acdc_conn_it]->isBoundryCancelable == false)
      {
        new_acdc.insert(*acdc_conn_it);
        continue;
      }

      if(msc->m_cps[*acdc_conn_it]->isCancelled == true)
        throw std::logic_error("*acdc_conn_it should not have been cancelled yet");

      id_t conn_cp_id = msc->m_cps[*acdc_conn_it]->cellid;

      cancel_pair_map_citer_t conn_cancel_pair_it
          = cancel_pair_map.find(conn_cp_id);

      if( conn_cancel_pair_it == cancel_pair_map.end())
      {
        std::stringstream ss;

        ss<<"missing canc pair in cancel_pair_map"<<std::endl;
        ss<<"cp pair = ("<<msc->m_cps[cp1_idx]->cellid<<","
                         <<msc->m_cps[cp2_idx]->cellid<<")"<<std::endl;

        throw std::logic_error(ss.str());
      }

      id_t conn_cp_pair_id  = conn_cancel_pair_it->second;

      if(msc->m_id_cp_map.find(conn_cp_pair_id) ==
         msc->m_id_cp_map.end())
        throw std::logic_error("missing second cancel pair element "\
                               "in msc->id_cp_map");

      uint conn_cp_pair_idx = msc->m_id_cp_map[conn_cp_pair_id];

      connection_t * conn_cp_acdc_conns[] =
      {&msc->m_cps[conn_cp_pair_idx]->asc,
       &msc->m_cps[conn_cp_pair_idx]->des};

      for(connection_iter_t conn_cp_acdc_conn_it =
          conn_cp_acdc_conns[i]->begin();
      conn_cp_acdc_conn_it != conn_cp_acdc_conns[i]->end();
      ++conn_cp_acdc_conn_it)
      {
        if(msc->m_cps[*conn_cp_acdc_conn_it]->isBoundryCancelable == true)
          throw std::logic_error("the connected cancelable cp shold not "\
                                 " contain a cancelable cp in its connections");

        new_acdc.insert(*conn_cp_acdc_conn_it);
      }
    }

    acdc_conns[i]->clear();
    acdc_conns[i]->insert(new_acdc.begin(),new_acdc.end());
  }

  msc->m_cps[cp1_idx]->isCancelled = false;
  msc->m_cps[cp2_idx]->isCancelled = false;
}

template <typename id_t>
    void uncancel_pairs
    ( MSComplex<id_t>  *msc,
      uint cp1_idx,
      uint cp2_idx)
{

  typedef typename MSComplex<id_t>::critical_point critical_point_t;
  typedef typename MSComplex<id_t>::critical_point::connection_t connection_t;
  typedef typename connection_t::iterator connection_iter_t;
  typedef typename MSComplex<id_t>::id_cp_map_t id_cp_map_t;
  typedef typename MSComplex<id_t>::cp_ptr_list_t cp_ptr_list_t;
  typedef typename std::pair<id_t,id_t> cancel_pair_t;
  typedef typename std::vector<cancel_pair_t> cancel_pair_list_t;

  if( msc->m_cps[cp1_idx]->asc.find ( cp2_idx ) ==
      msc->m_cps[cp1_idx]->asc.end() )
  {
    std::swap ( cp1_idx,cp2_idx );
  }

  connection_t * acdc_conns[] =
  {&msc->m_cps[cp1_idx]->asc,
   &msc->m_cps[cp2_idx]->des};

  uint * cp_idxs[]= {&cp1_idx,&cp2_idx};


  for(uint i = 0 ;i < 2 ; ++i)
  {

    if( acdc_conns[i]->find ( *cp_idxs[(i+1)%2] ) ==
        acdc_conns[i]->end())
    {
      throw std::logic_error("cancellable pair is not connected");
    }

    connection_t new_acdc;

    for(connection_iter_t acdc_conn_it = acdc_conns[i]->begin();
        acdc_conn_it != acdc_conns[i]->end();++acdc_conn_it)
    {
      if(*acdc_conn_it == *cp_idxs[(i+1)%2])
        continue;

      if(msc->m_cps[*acdc_conn_it]->isBoundryCancelable == false)
      {
        new_acdc.insert(*acdc_conn_it);
        continue;
      }

      if(msc->m_cps[*acdc_conn_it]->isCancelled == true)
        throw std::logic_error("*acdc_conn_it should not have been cancelled yet");


      uint conn_cp_pair_idx = msc->m_cps[*acdc_conn_it]->pair_idx;

      if(msc->m_cps[conn_cp_pair_idx]->pair_idx != *acdc_conn_it)
        throw std::logic_error("*acdc_conn_it and its pair dont agree on pairing");

      connection_t * conn_cp_acdc_conns[] =
      {&msc->m_cps[conn_cp_pair_idx]->asc,
       &msc->m_cps[conn_cp_pair_idx]->des};

      for(connection_iter_t conn_cp_acdc_conn_it =
          conn_cp_acdc_conns[i]->begin();
      conn_cp_acdc_conn_it != conn_cp_acdc_conns[i]->end();
      ++conn_cp_acdc_conn_it)
      {
        if(msc->m_cps[*conn_cp_acdc_conn_it]->isBoundryCancelable == true)
          throw std::logic_error("the connected cancelable cp shold not "\
                                 " contain a cancelable cp in its connections");

        new_acdc.insert(*conn_cp_acdc_conn_it);
      }
    }

    acdc_conns[i]->clear();
    acdc_conns[i]->insert(new_acdc.begin(),new_acdc.end());
  }

  msc->m_cps[cp1_idx]->isCancelled = false;
  msc->m_cps[cp2_idx]->isCancelled = false;
}

template <typename id_t>
  void form_cancel_pairs_map
  ( const std::vector<std::pair<id_t,id_t> > &cancel_pairs,
    std::map<id_t,id_t> &cancel_pair_map)
{
  std::copy(cancel_pairs.begin(),cancel_pairs.end(),
            std::inserter (cancel_pair_map, cancel_pair_map.begin()));

  std::transform(cancel_pairs.begin(),cancel_pairs.end(),
                 std::inserter(cancel_pair_map,cancel_pair_map.begin()),
                 reverse_pair<id_t,id_t>);
}



template <typename id_t>
    void union_complexes_down
    ( MSComplex<id_t>  *msc1,
      MSComplex<id_t>  *msc2 ,
      const std::vector<std::pair<id_t,id_t> > &cancellable_pairs,
      const std::map<id_t,id_t>  &cancel_pair_map)
{
  typedef typename MSComplex<id_t>::critical_point critical_point_t;
  typedef typename MSComplex<id_t>::critical_point::connection_t connection_t;
  typedef typename connection_t::iterator connection_iter_t;
  typedef typename MSComplex<id_t>::id_cp_map_t id_cp_map_t;
  typedef typename id_cp_map_t::iterator id_cp_map_iter_t;
  typedef typename MSComplex<id_t>::cp_ptr_list_t cp_ptr_list_t;
  typedef typename std::pair<id_t,id_t> cancel_pair_t;
  typedef typename std::vector<cancel_pair_t> cancel_pair_list_t;
  typedef typename cancel_pair_list_t::const_reverse_iterator cancel_pair_list_riter_t;
  typedef typename std::map<id_t,id_t> cancel_pair_map_t;
  typedef typename cancel_pair_map_t::iterator cancel_pair_map_iter_t;

  // carry out the uncancellation operation on msc1
  // in reverse order of the cancellation pairs

  for(cancel_pair_list_riter_t canpr_it = cancellable_pairs.rbegin();
  canpr_it != cancellable_pairs.rend() ; ++canpr_it)
  {

    id_t cp1_id = canpr_it->first;
    id_t cp2_id = canpr_it->second;

    if(msc1->m_id_cp_map.find(cp1_id) ==
       msc1->m_id_cp_map.end())
      throw std::logic_error("could not find cancelable cp1 in msc1");

    if(msc1->m_id_cp_map.find(cp2_id) ==
       msc1->m_id_cp_map.end())
      throw std::logic_error("could not find cancelable cp1 in msc2");

    uint cp1_idx = msc1->m_id_cp_map[cp1_id];
    uint cp2_idx = msc1->m_id_cp_map[cp2_id];

    uncancel_pairs(msc1,cp1_idx,cp2_idx,cancel_pair_map);
  }

  // now identify the uncancelled cps with msc2 and copy over the correct
  // connection info.. clear out the old that was there in the first place.
  // create new cp's for any new incidences.
  for(cancel_pair_list_riter_t canpr_it = cancellable_pairs.rbegin();
      canpr_it != cancellable_pairs.rend() ; ++canpr_it)
  {
    id_t cp1_id = canpr_it->first;
    id_t cp2_id = canpr_it->second;

    // the verification of availability of the id's in msc1
    // was already done in the last loop

    uint msc1_cp1_idx = msc1->m_id_cp_map[cp1_id];
    uint msc1_cp2_idx = msc1->m_id_cp_map[cp2_id];

    if(msc2->m_id_cp_map.find(cp1_id) ==
       msc2->m_id_cp_map.end())
      throw std::logic_error("could not find cancelable cp1 in msc1");

    if(msc2->m_id_cp_map.find(cp2_id) ==
       msc2->m_id_cp_map.end())
      throw std::logic_error("could not find cancelable cp2 in msc2");

    uint msc2_cp1_idx = msc2->m_id_cp_map[cp1_id];
    uint msc2_cp2_idx = msc2->m_id_cp_map[cp2_id];

    connection_t * msc1_acdc_conns[] =
    {&msc1->m_cps[msc1_cp1_idx]->asc,&msc1->m_cps[msc1_cp1_idx]->des,
     &msc1->m_cps[msc1_cp2_idx]->asc,&msc1->m_cps[msc1_cp2_idx]->des};

    connection_t * msc2_acdc_conns[] =
    {&msc2->m_cps[msc2_cp1_idx]->asc,&msc2->m_cps[msc2_cp1_idx]->des,
     &msc2->m_cps[msc2_cp2_idx]->asc,&msc2->m_cps[msc2_cp2_idx]->des};

    for(uint i= 0 ; i < 4; ++i)
    {
      msc2_acdc_conns[i]->clear();

      for(connection_iter_t conn_cp_it = msc1_acdc_conns[i]->begin();
          conn_cp_it != msc1_acdc_conns[i]->end(); ++conn_cp_it)
      {
        id_t conn_cp_id = msc1->m_cps[*conn_cp_it]->cellid;

        if(msc1->m_cps[*conn_cp_it]->isBoundryCancelable == true)
          throw std::logic_error("expected only non bc cps");

        critical_point_t* src_cp = msc1->m_cps[*conn_cp_it];

        if(msc2->m_id_cp_map.find(conn_cp_id) ==
           msc2->m_id_cp_map.end())
        {
          // allocate a new cp
          critical_point_t* dest_cp = new critical_point_t;

          // copy over the trivial data
          dest_cp->isCancelled               = src_cp->isCancelled;
          dest_cp->isBoundryCancelable       = src_cp->isBoundryCancelable;
          dest_cp->isOnStrangulationPath     = src_cp->isOnStrangulationPath;
          dest_cp->cellid                    = src_cp->cellid;

          msc2->m_id_cp_map[dest_cp->cellid] = msc2->m_cps.size();
          msc2->m_cps.push_back(dest_cp);
        }

        uint msc2_conn_cp_idx = msc2->m_id_cp_map[conn_cp_id];

        msc2_acdc_conns[i]->insert(msc2_conn_cp_idx);
      }
    }
  }

  // finally for each non bc cp in msc2 adjust connections such that the
  // the bc cp's are cleared out and any connections with the updated list
  // of cps is reflected

  for(uint i = 0 ; i < msc2->m_cps.size();++i)
  {
    critical_point_t * msc2_cp = msc2->m_cps[i];

    if(msc2_cp->isBoundryCancelable == true)
      continue;

    id_cp_map_iter_t msc1_id_cp_it = msc1->m_id_cp_map.find(msc2_cp->cellid);

    if(msc1_id_cp_it == msc1->m_id_cp_map.end())
      throw std::logic_error("missing non bc cp in msc1 present in msc2");

    critical_point_t *msc1_cp = msc1->m_cps[msc1_id_cp_it->second];

    connection_t * msc1_cp_acdc_conns[] = {&msc1_cp->asc,&msc1_cp->des};
    connection_t * msc2_cp_acdc_conns[] = {&msc2_cp->asc,&msc2_cp->des};

    for(uint j = 0 ; j<2; ++j)
    {
      msc2_cp_acdc_conns[j]->clear();

      for(connection_iter_t conn_it = msc1_cp_acdc_conns[j]->begin();
          conn_it != msc1_cp_acdc_conns[j]->end(); ++conn_it)
      {
        id_t conn_cp_id = msc1->m_cps[*conn_it]->cellid;

        id_cp_map_iter_t msc2_conn_cp_it = msc2->m_id_cp_map.find(conn_cp_id);

        if(msc2_conn_cp_it == msc2->m_id_cp_map.end())
          continue;

        msc2_cp_acdc_conns[j]->insert(msc2_conn_cp_it->second);

      }
    }
  }
}

template <typename id_t>
void print_connections
(std::ostream & os,
 const MSComplex<id_t> &msc,
 const typename MSComplex<id_t>::critical_point::connection_t &conn
 )
{
  typedef typename MSComplex<id_t>::critical_point::connection_t connection_t;
  typedef typename connection_t::iterator conn_iter_t;

  os<<"{ ";
  for(conn_iter_t it = conn.begin(); it != conn.end(); ++it)
  {
    if(msc.m_cps[*it]->isBoundryCancelable)
      os<<"*";
    os<<msc.m_cps[*it]->cellid;
    os<<", ";
  }
  os<<"}";
}

template <typename id_t>
void print_connections
(std::ostream & os,const MSComplex<id_t> &msc)
{
  for(uint i = 0 ; i < msc.m_cps.size();++i)
  {
    os<<"des(";
    if(msc.m_cps[i]->isBoundryCancelable)
      os<<"*";
    os<<msc.m_cps[i]->cellid<<") = ";
    print_connections(os,msc,msc.m_cps[i]->des);
    os<<std::endl;

    os<<"asc(";
    if(msc.m_cps[i]->isBoundryCancelable)
      os<<"*";
    os<<msc.m_cps[i]->cellid<<") = ";
    print_connections(os,msc,msc.m_cps[i]->asc);
    os<<std::endl;
    os<<std::endl;
  }
}



template<typename id_t>
    QuadGenericCell2D<id_t> createQuadGenericCell2D ( const id_t *v,const unsigned int &num_v,const unsigned char &dim )
{
  QuadGenericCell2D<id_t> gen_cell;

  for ( uint i = 0 ;i<num_v;i++ )
  {
    gen_cell._v[i] = v[i];
  }

  std::sort ( gen_cell._v,gen_cell._v+num_v );

  std::reverse ( gen_cell._v,gen_cell._v+num_v );

  for ( uint i = num_v ;i<4;i++ )
  {
    gen_cell._v[i] = gen_cell._v[num_v-1];
  }

  gen_cell._dim = dim;

  return gen_cell;
}

template<typename id_t>
    TriGenericCell2D<id_t> createTriGenericCell2D ( const id_t *v,const unsigned int &num_v,const unsigned char &dim )
{
  TriGenericCell2D<id_t> gen_cell;

  for ( uint i = 0 ;i<num_v;i++ )
  {
    gen_cell._v[i] = v[i];
  }

  std::sort ( gen_cell._v,gen_cell._v+num_v );

  std::reverse ( gen_cell._v,gen_cell._v+num_v );

  for ( uint i = num_v ;i<3;i++ )
  {
    gen_cell._v[i] = gen_cell._v[num_v-1];
  }

  gen_cell._dim = dim;

  return gen_cell;
}

#endif

