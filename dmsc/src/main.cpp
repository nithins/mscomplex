#include <exception>
#include <string>

#include <boost/regex.hpp>

#include <logutil.h>

#include <framework.h>
#include <quad_datamanager.h>
#include <grid_datamanager.h>

using namespace std;

string usage_string(const char * progname)
{
  stringstream ss;

  ss<<"Usage: ";

  ss<<progname<<" ";

  ss<<"-<qf/gf> <filename> -d <x_dim> <y_dim> "\
      "[--max-levels <max-levels>] "\
      "[--buffer-zone-width <buf-zone-width>]"\
      "[-st]"<<endl;

  return ss.str();
}

IModel * parse_quad_grid(string cmdline)
{

  const boost::regex file_re ( "(-qf ([[:alnum:]\\./]+))" );

  const boost::regex dim_re ( "(-d ([[:digit:]]+) ([[:digit:]]+))" );

  const boost::regex max_levels_re ( "(--max-levels ([[:digit:]]+))" );

  const boost::regex buffer_zone_re ( "(--buffer-zone-width ([[:digit:]]+))" );

  const boost::regex st_re ( "(-st)" );

  uint   size_x = 0, size_y = 0;

  string filename;

  uint   max_levels  = 1;

  uint   buf_zone_size = 1;

  bool   single_thread = false;

  boost::smatch matches;

  if ( regex_search ( cmdline,matches, file_re ) )
  {
    filename.assign ( matches[2].first, matches[2].second );
  }
  else
  {
    _LOG_FILE_N_FUNC();
    throw invalid_argument("No File name specified");
  }

  if ( regex_search ( cmdline,matches, dim_re ) )
  {
    string sx ( matches[2].first,matches[2].second );
    string sy ( matches[3].first,matches[3].second );

    size_x = atoi ( sx.c_str() );
    size_y = atoi ( sy.c_str() );
  }
  else
  {
    _LOG_FILE_N_FUNC();
    throw invalid_argument("No Dim specified");
  }

  if ( regex_search ( cmdline,matches, st_re ) )
  {
    single_thread = true;
  }

  if ( regex_search ( cmdline,matches, max_levels_re ) )
  {
    string ml ( matches[2].first,matches[2].second );

    max_levels = atoi ( ml.c_str() );
  }

  if ( regex_search ( cmdline,matches, buffer_zone_re ) )
  {
    string bz ( matches[2].first,matches[2].second );

    buf_zone_size = atoi ( bz.c_str() );
  }

  return new QuadDataManager(filename,size_x,size_y,buf_zone_size,max_levels,single_thread);
}

IModel * parse_grid(string cmdline)
{

  const boost::regex file_re ( "(-gf ([[:alnum:]\\./]+))" );

  const boost::regex dim_re ( "(-d ([[:digit:]]+) ([[:digit:]]+))" );

  const boost::regex st_re ( "(-st)" );

  uint   size_x = 0, size_y = 0;

  string filename;

  bool   single_thread = false;

  boost::smatch matches;

  if ( regex_search ( cmdline,matches, file_re ) )
  {
    filename.assign ( matches[2].first, matches[2].second );
  }
  else
  {
    _LOG_FILE_N_FUNC();
    throw invalid_argument("No File name specified");
  }

  if ( regex_search ( cmdline,matches, dim_re ) )
  {
    string sx ( matches[2].first,matches[2].second );
    string sy ( matches[3].first,matches[3].second );

    size_x = atoi ( sx.c_str() );
    size_y = atoi ( sy.c_str() );
  }
  else
  {
    _LOG_FILE_N_FUNC();
    throw invalid_argument("No Dim specified");
  }

  if ( regex_search ( cmdline,matches, st_re ) )
  {
    single_thread = true;
  }

  return new GridDataManager(filename,size_x,size_y,single_thread);
}


int main ( int argc, char *argv[] )
{

  const boost::regex qf_re ( "(-qf )");
  const boost::regex gf_re ( "(-gf )");


//  try
  {
    boost::shared_ptr<IFramework> framework(IFramework::Create ( argc, argv ));

    std::stringstream ss;

    for ( int i = 1 ; i < argc;i++ )
      ss << argv[i] << " ";

    string cmdline = ss.str();

    boost::shared_ptr<IModel> model;

    if(regex_search ( cmdline, qf_re ))
    {
      model.reset(parse_quad_grid(cmdline));

    }
    else  if(regex_search ( cmdline, gf_re ))
    {
      model.reset(parse_grid(cmdline));
    }
    else
    {
      throw std::invalid_argument("no data type specified");
    }

    framework->AddModel ( model );

    framework->Exec();
  }
//  catch(std::exception &e)
//  {
//    _LOG(usage_string(argv[0]));
//    _LOG(e.what());
//  }

  return 0;
}
