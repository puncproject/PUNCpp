#include "injector.h"
#include "population.h"

double erfinv(const double &P)
{
  double Y, A, B, X, Z, W, WI, SN, SD, F, Z2, SIGMA;
  double A1 = -.5751703, A2 = -1.896513, A3 = -.5496261E-1;
  double B0 = -.1137730, B1 = -3.293474, B2 = -2.374996, B3 = -1.187515;
  double C0 = -.1146666, C1 = -.1314774, C2 = -.2368201, C3 = .5073975e-1;
  double D0 = -44.27977, D1 = 21.98546, D2 = -7.586103;
  double E0 = -.5668422E-1, E1 = .3937021, E2 = -.3166501, E3 = .6208963E-1;
  double F0 = -6.266786, F1 = 4.666263, F2 = -2.962883;
  double G0 = .1851159E-3, G1 = -.2028152E-2, G2 = -.1498384, G3 = .1078639E-1;
  double H0 = .9952975E-1, H1 = .5211733, H2 = -.6888301E-1;
  // double RINFM=1.7014E+38;

  X = P;
  SIGMA = copysign(1.0, X);
  //it_error_if(X < -1 || X > 1, "erfinv : argument out of bounds");
  Z = fabs(X);
  if (Z > .85) {
    A = 1 - Z;
    B = Z;
    W = sqrt(-log(A + A * B));
    if (W >= 2.5) {
      if (W >= 4.) {
        WI = 1. / W;
        SN = ((G3 * WI + G2) * WI + G1) * WI;
        SD = ((WI + H2) * WI + H1) * WI + H0;
        F = W + W * (G0 + SN / SD);
      }
      else {
        SN = ((E3 * W + E2) * W + E1) * W;
        SD = ((W + F2) * W + F1) * W + F0;
        F = W + W * (E0 + SN / SD);
      }
    }
    else {
      SN = ((C3 * W + C2) * W + C1) * W;
      SD = ((W + D2) * W + D1) * W + D0;
      F = W + W * (C0 + SN / SD);
    }
  }
  else {
    Z2 = Z * Z;
    F = Z + Z * (B0 + A1 * Z2 / (B1 + Z2 + A2 / (B2 + Z2 + A3 / (B3 + Z2))));
  }
  Y = SIGMA * F;
  return Y;
}

double erfcinv(const double &P)
{
    return erfinv(1.0-P);
}

RSND::RSND(std::function<double (std::vector<double>&)> f, const double& f_max, 
           const std::vector<double>& L):pdf(f), pdf_max(f_max), Ld(L), 
           dim(L.size()), dist(0.0, f_max), rng{random_seed_seq::get_instance()}
{
    std::vector<std::uniform_real_distribution<double>> dists(dim);
    for (i = 0; i<dim; ++i)
    {
        dists[i] = distribution(0.0, Ld[i]);
    }
    this->dists = dists;
        
}

void RSND::generate_v(std::vector<double>& v)
{
    for (i=0; i<dim; ++i)
    {
        v[i] = dists[i](rng);
    }
}

void RSND::generate_vs(int& n, std::vector<double>& v, std::vector<double>& vs)
{
    if (dist(rng) < pdf(v))
    {
        // for (i = 0; i<dim; ++i)
        // {
        //     vs[i*vs.size()/dim + n] = v[i];
        // }
        for (i = n*dim; i<(n+1)*dim; ++i)
        {
            vs[i] = v[i%dim];
        }
        n += 1;
    }
}

std::vector<double> RSND::sample(const int &N)
{
    std::vector<double> vs(N*dim), v(dim);
    int n = 0;
    while (n < N)
    {
        generate_v(v);
        generate_vs(n, v, vs);
    }
    return vs;
}

RS1D::RS1D(){};

RS1D::RS1D(std::function<double (double&)> f, double f_max, 
           std::function<double (double&)> g, double lb, double ub):pdf(f), 
           pdf_max(f_max), transform(g), lb(lb), ub(ub), dist(0.0, pdf_max),
           dists(lb, ub), rng{random_seed_seq::get_instance()}{};

void RS1D::generate_v(std::vector<double>& v)
{
    for (i = 0; i < v.size(); ++i)
    {
        v[i] = dists(rng);
    }
}

void RS1D::generate_vs(int& n, std::vector<double>& v, std::vector<double>& vs)
{
    for (i = 0; i < v.size(); ++i)
    {
        if (dist(rng) < pdf(v[i]))
        {
            vs[n] = transform(v[i]);
            n += 1; 
        }
    }
}

std::vector<double> RS1D::sample(const int &N)
{
    std::vector<double> vs(N);
    std::vector<double> v(N);
    n = 0;

    while (n < N)
    {
        int m = N-n;
        v.resize(m);
        generate_v(v);
        generate_vs(n, v, vs);
    }
    return vs;
}

ORS::ORS(){};

ORS::ORS(std::function<double (double&)> f, std::function<double (double&)> df, 
         std::vector<double>& roots, std::function<double (double&)> g, 
         std::vector<double> interval, int nsp):Vt(f), dVdt(df), 
         roots(roots), transform(g), lb(interval[0]), ub(interval[1]), nsp(nsp), 
         dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
{          
    std::vector<double> points;
    auto dVroot = 1;
    while (dVroot != 0)
    {
        dVroot = 0;
        points = linspace(lb, ub, nsp+2);
        for(auto &ps: points)
        {
            for(auto &rs: roots)
            {
                if (ps==rs)
                {
                    dVroot += 1;
                    nsp += 1;
                }
            }
        }
    }
    std::vector<double> sp(nsp), y(nsp), dy(nsp);
    for ( i = 0; i < nsp; ++i )
    {
        sp[i] = points[i+1];
        y[i]  = Vt(sp[i]);
        dy[i] = dVdt(sp[i]);
    }
    this->sp  = sp;
    this->nsp = nsp;
    this->y   = y;
    this->dy  = dy;

    lower_hull();

    std::vector<double> integrand(nsp);
    int num_negative{0};
    for ( i = 0; i < nsp; ++i)
    {
        integrand[i] = (exp(-Yt[i+1]) - exp(-Yt[i]))/(-dy[i]);
        if (integrand[i] < 0.0)
        {
            num_negative += 1;
        }
    }
    while (num_negative > 0)
    {
        nsp += 1;
        points = linspace(lb, ub, nsp+2);
        sp.resize(nsp);
        y.resize(nsp);
        dy.resize(nsp);
        for ( i = 0; i < nsp; ++i )
        {
            sp[i] = points[i+1];
            y[i]  = Vt(sp[i]);
            dy[i] = dVdt(sp[i]);
        }
        this->sp  = sp;
        this->nsp = nsp;
        this->y   = y;
        this->dy  = dy;
        
        lower_hull();

        integrand.resize(nsp);
        num_negative = 0;
        for ( i = 0; i < nsp; ++i)
        {
            integrand[i] = (exp(-Yt[i+1]) - exp(-Yt[i]))/(-dy[i]);
            if (integrand[i] < 0.0)
            {
                num_negative += 1;
            }
        }
    }
    this->integrand = integrand;
    exponentiated_lower_hull();
}

void ORS::tangent_intersections()
{
    std::vector<double> z(nsp+1);
    z[0]   = lb;
    z[nsp] = ub;
    for( i = 1; i < nsp; ++i)
    {
        z[i] = ( y[i] - y[i-1] - sp[i]*dy[i] + sp[i-1]*dy[i-1])/(dy[i-1]-dy[i]);
    }
    this->z = z;
}

void ORS::lower_hull()
{
    tangent_intersections();
    std::vector<double> Yt(nsp+1);
    Yt[0] = y[0] + dy[0]*(z[0] - sp[0]);
    for ( i = 0; i < nsp; ++i)
    {
        Yt[i+1] = y[i] + dy[i]*(z[i+1] - sp[i]);
    }
    this->Yt = Yt;
}

void ORS::exponentiated_lower_hull()
{
    std::vector<double> exp_cdf(nsp+1);
    exp_cdf[0] = 0;
    for( i = 0; i < nsp; ++i)
    {
        exp_cdf[i+1] = exp_cdf[i] + integrand[i];
    }
    this->exp_cdf = exp_cdf;
    this->c_i     = exp_cdf[nsp];
}

std::vector<double> ORS::sample_exp(const int &N)
{
    std::vector<double> t(2*N);
    int j = 0;
    double r;
    while (j < N)
    {
        r = c_i*dist(rng);
        for (i=0; i<nsp-1; ++i)
        {
            if ((exp_cdf[i])<r && (exp_cdf[i+1])>r)
            {
                t[2*j+1] = (double)i;
                t[2*j]   = sp[i] + (y[i]+log(-dy[i]*(r - exp_cdf[i]) +\
                           exp(-Yt[i]))) / (-dy[i]);
                j += 1;
                break;
            } 
        }
    }
    return t;
}

std::vector<double> ORS::sample(const int &N)
{
    std::vector<double> vs(N);
    n = 0;
    int m, j, index;
    while (n < N)
    {
        m = N-n;
        std::vector<double> t(2*m), y_t(m), Y_i(m);
        t = sample_exp(m);
 
        for ( j = 0; j < m; ++j)
        {
            index  = (int)(t[2*j+1]);
            y_t[j] = Vt(t[2*j]);
            Y_i[j] = y[index] + (t[2*j]-sp[index])*dy[index];
            if( dist(rng) < exp(Y_i[j]-y_t[j]))
            {
                vs[n] = transform(t[2*j]);
                n++;
            }
        } 
    }
    return vs;
}

// Maxwellian::Maxwellian(){};

Maxwellian::Maxwellian(const std::vector<double> v_drift, 
                       double v_thermal, std::vector<bool> periodic, 
                       double vd_ratio): vd(v_drift), vth(v_thermal), 
                       periodic(periodic), dim(v_drift.size()), vd_ratio(vd_ratio)
{
    if (v_thermal == 0.0)
    {
        this->vth = std::numeric_limits<double>::epsilon();
    }
    int d = 0;
    for(auto &&p: periodic)
    {
        if(!p)
        {
            d++;
        }
    }
    this->dim_nonperiodic = d;

    initialize_loading();
    initialize_injection();
};

void Maxwellian::initialize_loading()
{
    int i;
    std::vector<std::function<std::vector<double>(int)>> loader(dim);
    for(i=0; i<dim; ++i)
    {
        if(vd[i]==0)
        {
            auto tmp0 = cdf_inv_nondrifting();
            loader[i] = [this, tmp0](int N){return generate(tmp0,N);};
        } else
        {
            auto tmp1 = cdf_inv_drifting(i);
            loader[i] = [this, tmp1](int N){return generate(tmp1,N);};
        }
    }
    this->loader = loader;
}

void Maxwellian::initialize_injection()
{
    if (dim_nonperiodic != 0)
    {
        int i, j;
        this->dim2 = dim*dim;
        this->len_cdf = dim*dim_nonperiodic;

        std::vector<bool> normal_vec(dim2, false);
        j = 0;
        for (i = 0; i < dim; ++i)
        {
            normal_vec[j+ i*dim] = true;
            j++;
        }
        this->normal_vec = normal_vec;

        std::vector<double> vd_nonzero;
        j = 0;
        for(i=0; i<dim; ++i)
        {
            if (vd[i]!=0)
            {
                vd_nonzero.push_back(vd[i]);
                j += 1;
            }
        }
        int dim_drift = j;
        this->dim_drift = dim_drift;
        this->vd_nonzero = vd_nonzero;

        if(dim_drift==0.0)
        {
            inject_nondrifting();
        } else
        {
            initialize_ors();
            inject_drifting();
        }
    }
}

void Maxwellian::initialize_ors()
{
    std::vector<ORS> rs(2*dim_drift);

    std::vector<double> s{1., -1.}; 
        
    double vd_n;
    int i, j;

    for (i = 0; i<dim_drift; ++i)
    {
        vd_n = vd_nonzero[i]/vth;
        j = 0;
        for (auto & si: s)
        {   
            std::vector<double> root;
            root.push_back(pdf_max_drifting(vd_n, si));

            auto V = [vd_n, si](double& t){ return -log(t) + \
            3.*log(1.-t) + 0.5*(vd_n - si*t/(1.-t))*(vd_n - si*t/(1.-t));};
            
            auto dV = [vd_n, si](double& t){ return (-1./t) - \
            (3./(1.-t)) + (si*(t*(si+vd_n)-vd_n)/((1.-t)*(1.-t)*(1.-t)));};
            
            auto transform = [this, si](double& t){return si*vth*t/(1.-t);};
            
            ORS ors(V, dV, root, transform);
            rs[i+j*dim_drift] = ors;
            j += 1;
        }
    }
    this->rs = rs;    
}

double Maxwellian::pdf_max_drifting(const double vd_n, const int s)
{
    double ca = 2.0;
    double cb = -4.0 - s*vd_n;
    double cc = s*vd_n;
    double cd = 1.0;

    double p = (3.0*ca*cc - cb*cb)/(3.0*ca*ca);
    double q = (2.0*cb*cb*cb - 9.0*ca*cb*cc + 27*ca*ca*cd) / (27*ca*ca*ca);

    double sqrt_term = sqrt(-p/3.0);
    double root = 2.0*sqrt_term*\
           cos(acos(3.0*q/(2.0*p*sqrt_term))/3.0 - 2.0*M_PI/3.0);
   double tmp = root - cb/(3.0*ca);
   return tmp;
}

std::function<double(double&)> Maxwellian::pdf_flux_drifting_transformed(
                                      const double vd_n, const int s)
{
    auto pdf = [vd_n, s](double& t){
        return s*t*exp(-.5*(t/(s*(1.-t))-vd_n)*(t/(s*(1.-t))-vd_n))/\
        ((s*(1.-t))*(s*(1.-t))*(s*(1.-t)));};
    return pdf;
}

std::function<double(double&)> Maxwellian::pdf_maxwellian(const int index)
{
    auto pdf = [this, index](double& t){
        return 1.0/(sqrt(2.*M_PI)*vth)*\
               exp(-0.5*((t-vd[index])*(t-vd[index]))/\
               (vth*vth));
    };
    return pdf;
}

std::function<double(double&)> Maxwellian::pdf_flux_nondrifting(const int s)
{
    auto pdf = [this, s](double& t){
        return s*(t/(vth*vth))*exp(-.5*(t/vth)*(t/vth));
    };
    return pdf;
}

std::function<double(double&)> Maxwellian::pdf_flux_drifting(const int index, 
                                                        const int s)
{
    auto pdf = [this, index, s](double& t){
        return (s*t*exp(-0.5*((t-vd[index])*(t-vd[index]))/(vth*vth)))/\
               (vth*vth*exp(-0.5*(vd[index]/vth)*(vd[index]/vth)) +\
               s*sqrt(0.5*M_PI)*vd[index]*vth * \
               (1. + s*erf(vd[index]/(sqrt(2.)*vth))));
    };
    return pdf;
}

std::function<double(double&)> Maxwellian::cdf_inv_nondrifting()
{
    auto cdf = [this](double& t){
               return sqrt(2.0)*vth*erfinv(2.*t-1.0);};
    return cdf;
}

std::function<double(double&)> Maxwellian::cdf_inv_drifting(const int k)
{
    auto cdf = [this, k](double& t){
               return vd[k]-sqrt(2.0)*vth*erfcinv(2*t);};
    return cdf;
}

std::function<double(double&)> Maxwellian::cdf_inv_flux_drifting(const int index)
{
    double m = (vd[index]*vd[index]+vth*vth)/vd[index];
    auto cdf = [this, m](double& t){return m + sqrt(2.)*vth*erfinv(2.*t-1.);};
    return cdf;
}

std::function<double(double&)> Maxwellian::cdf_inv_inward_flux_nondrifting()
{
    auto cdf = [this](double& t){
        return vth*sqrt(-2.*log(1.0-t));};
    return cdf;
}

std::function<double(double&)> Maxwellian::cdf_inv_backward_flux_nondrifting()
{
    auto cdf = [this](double& t){
        return -vth*sqrt(-2.0*log(t));};
    return cdf;
}

void Maxwellian::inject_nondrifting()
{
    auto maxwellian_flux = [this](int N){
                                    return generate(cdf_inv_nondrifting(), N);};
    auto inward_flux = [this](int N){
                        return generate(cdf_inv_inward_flux_nondrifting(), N);};
    auto backward_flux = [this](int N){
                      return generate(cdf_inv_backward_flux_nondrifting(), N);};  

    int i, j, k;
    std::vector<std::function<std::vector<double> (int)>> cdf(2*dim2);
    std::vector<std::function<std::vector<double> (int)>> generator(2*len_cdf);
    for (i = 0; i < dim; ++i)
    {
        for (j = 0; j < dim; ++j)
        {
            k = i*dim + j;
            if(normal_vec[k])
            {
                cdf[k] = inward_flux;
                cdf[k+dim2] = backward_flux;
            }else{
                cdf[k] = maxwellian_flux;
                cdf[k+dim2] = maxwellian_flux;                
            }
        }
    }

    k = 0;
    for (i = 0; i < dim; ++i)
    {
        if(!periodic[i])
        {
            for(j = (i*dim); j < ((i+1)*dim); ++j)
            {
                generator[k] = cdf[j];
                generator[k+len_cdf] = cdf[j+dim2];
                k++;
            }
        }
    }
    this->generator = generator;
}

void Maxwellian::inject_drifting()
{
    int i, j, k, m;

    std::vector<std::function<std::vector<double>(int)>> cdf(2*dim2);
    std::vector<std::function<std::vector<double>(int)>> generator(2*len_cdf);

    m = 0;
    for (i = 0; i < dim; i++)
    {
        for (j = 0; j < dim; j++)
        {
            k = i*dim +j;

            if(vd[j]!=0)
            {
                if(normal_vec[k])
                {
                    if (vd[i]/vth < vd_ratio)
                    {
                        cdf[k] = [this,m](const int N){return rs[m].sample(N);};
                    }else{
                        cdf[k] = [this,k](const int N)
                            {return generate(cdf_inv_flux_drifting(k%dim), N);};
                    }
                    cdf[k+dim2] = [this, m](const int N)
                                           { return rs[m+dim_drift].sample(N);};
                    m++;
                }else
                {
                    cdf[k]    = [this, k](const int N){
                                return generate(cdf_inv_drifting(k%dim), N);};
                    cdf[k+dim2] = [this, k](const int N){
                                return generate(cdf_inv_drifting(k%dim), N);};
                }
            }else
            {
                if(normal_vec[k])
                {
                    cdf[k]    = [this](const int N){
                        return generate(cdf_inv_inward_flux_nondrifting(), N);};
                    cdf[k+dim2] = [this](const int N){ 
                      return generate(cdf_inv_backward_flux_nondrifting(), N);};
                }else
                {
                    cdf[k]    = [this](const int N){
                                return generate(cdf_inv_nondrifting(), N);};
                    cdf[k+dim2] = [this](const int N){
                                return generate(cdf_inv_nondrifting(), N);};
                }
            }
        }
    }
    k = 0;
    for (i=0; i<dim; i++)
    {
        if(!periodic[i])
        {
            for(j=(i*dim); j<((i+1)*dim); j++)
            {
                generator[k] = cdf[j];
                generator[k+len_cdf] = cdf[j+dim2];
                k++;
            }
        }
    }
    this->generator = generator; 
}

std::vector<double> Maxwellian::generate(std::function<double(double&)> cdf_inv, int N)
{
    int i;
    std::vector<double> vs(N);
    for (i = 0; i < N; ++i)
    {
        r = dist(rng);
        vs[i] = cdf_inv(r);
    }
    return vs;
}

std::vector<double> Maxwellian::sample(const int &N, int k=0)
{
    int i, j;
    std::vector<double> vs(N*dim), v(N);
    for (j = 0; j < dim; ++j)
    {
        v = generator[j+k*dim](N);
        for (i = 0; i < N; ++i)
        {
            vs[i*dim+j] = v[i];
        }
    }
    return vs;
}

std::vector<double> Maxwellian::load(const int &N)
{
    int i, j;
    std::vector<double> vs(N*dim), v(N);
    for (j = 0; j < dim; ++j)
    {
        v = loader[j](N);
        for (i = 0; i < N; ++i)
        {
            vs[i*dim+j] = v[i];
        }
    }
    return vs;
}

std::vector<double> Maxwellian::flux_number(const double &plasma_density, 
                                    const std::vector<double> &surface_area, 
                                    const double &dt)
{
    auto backward_particle_flux_number = [this, plasma_density, surface_area,
    dt](int i){ return abs(plasma_density*surface_area[i%dim_nonperiodic]*dt*\
                (0.5*vd[i]*erfc(vd[i]/(sqrt(2.0)*vth)) -\
                vth/(sqrt(2*M_PI))*exp(-0.5*(vd[i]/vth)*(vd[i]/vth))));};

    auto inward_particle_flux_number = [this, plasma_density, surface_area,
    dt](int i){ return plasma_density*surface_area[i%dim_nonperiodic]*dt*\
                (vth/(sqrt(2*M_PI))*exp(-0.5*(vd[i]/vth)*(vd[i]/vth)) +\
                0.5*vd[i]*(1. + erf(vd[i]/(sqrt(2)*vth))));};

    std::vector<double> N(2*dim_nonperiodic);
    int i, j = 0;
    for(i = 0; i < dim; ++i)
    {
        if(!periodic[i])
        {
            N[j] = inward_particle_flux_number(i);
            N[j+dim_nonperiodic] = backward_particle_flux_number(i);
            j++;
        }
    }
    return N;
}

MovePeriodic::MovePeriodic(Population& pop, double dt): pop(pop), Ld(pop.Ld), 
                           dim(pop.dim), dt(dt){};

void MovePeriodic::move()
{
    int i, j;
    for(i=0; i<pop.tot_num; ++i)
    {
        for(j=0; j<dim; ++j)
        {
            pop.xs[i*dim+j] += dt*pop.vs[i*dim+j];
            pop.xs[i*dim+j] -= Ld[j] * floor(pop.xs[i*dim+j]/Ld[j]);
        }
    }    
}

MoveNonPeriodic::MoveNonPeriodic(Population &pop, double dt): 
                                 pop(pop), Ld(pop.Ld), dim(pop.dim), dt(dt){};

void MoveNonPeriodic::move()
{
    std::vector<int> outside;
    int i, j;
    std::vector<double>& xs = pop.xs;
    std::vector<double>& vs = pop.vs;

    std::cout << "total before move:   " << pop.tot_num << '\n';

    int tmp = pop.tot_num;
    for(i=0; i<pop.tot_num; ++i)
    {
        for(j=0; j<dim; ++j)
        {

            xs[i*dim+j] += dt*vs[i*dim+j];

            if (xs[i*dim+j]<0.0 || xs[i*dim+j]>Ld[j])
            {
                outside.push_back(i);
                break;
            }
        }
    }

    int len = outside.size();
    this->count.push_back(len);
    std::cout << "outside: "<<len<<'\n';
    if(len>0)
    {
        for (i = len-1; i>=0; --i)
        {
            std::swap(pop.ids[outside[i]], pop.ids.back());
            pop.ids.pop_back();
        }
        for (j = dim-1; j >= 0; --j)
        {
            for (i = len-1; i >= 0; --i)
            {
                std::swap(xs[outside[i]*dim+j], xs.back());
                xs.pop_back();
                std::swap(vs[outside[i]*dim+j], vs.back());
                vs.pop_back();            
            }

        }
        pop.xs = xs;
        pop.vs = vs;    
        pop.tot_num = pop.xs.size()/dim;
    }
    double su = 0.0;
    for(auto& s: count)
    {
        su += s;
    }
    std::cout << "average: "<<su/count.size()<<'\n';
    std::cout << "total after move:   " << pop.tot_num<<", num outside: "<< tmp-pop.tot_num<< '\n'; 
}

MoveMixedBnd::MoveMixedBnd(Population& pop, double dt, 
                           std::vector<int> periodic_indices, 
                           std::vector<int> nonperiodic_indices): 
                           pop(pop), Ld(pop.Ld), periodic(pop.periodic), 
                           dim(pop.dim), dt(dt), 
                           periodic_indices(periodic_indices), 
                           nonperiodic_indices(nonperiodic_indices)
{

}
void MoveMixedBnd::move()
{
    std::vector<int> outside;
    int i;

    for(i=0; i<pop.tot_num; ++i)
    {
        for(auto& j: periodic_indices)
        {
            pop.xs[i*dim+j] += dt*pop.vs[i*dim+j];
            pop.xs[i*dim+j] -= Ld[j] * floor(pop.xs[i*dim+j]/Ld[j]);
        }
        for(auto& j: nonperiodic_indices)
        {
            pop.xs[i*dim+j] += dt*pop.vs[i*dim+j];
            if (pop.xs[i*dim+j]<0.0 || pop.xs[i*dim+j]>Ld[j])
            {
                outside.push_back(i);
                break;
            }
        }
    }

    int len = outside.size();
    if(len>0)
    {   
        int j;
        for (i = len-1; i>=0; --i)
        {
            std::swap(pop.ids[outside[i]], pop.ids.back());
            pop.ids.pop_back();
        }
        for (j = dim-1; j >= 0; --j)
        {
            for (i = len-1; i >= 0; --i)
            {
                std::swap(pop.xs[outside[i]*dim+j], pop.xs.back());
                pop.xs.pop_back();
                std::swap(pop.vs[outside[i]*dim+j], pop.vs.back());
                pop.vs.pop_back();            
            }

        }    
        pop.tot_num = pop.xs.size()/dim;
    }
}

Injector::Injector(Population& pop, double dt): pop(pop), dt(dt), 
                   periodic(pop.periodic), Ld(pop.Ld), 
                   plasma_density(pop.plasma_density), dim(Ld.size()), 
                   dist(0.0, 1.0), rng{random_seed_seq::get_instance()}
{
    double vth = pop.vth;
    std::vector<double> vd = pop.vd;
    this->mv = std::make_unique<Maxwellian>(vd, vth, periodic);

    initialize_injection();
}

void Injector::inject()
{
    
    int i, j, k, N, n, num;
    double Nd, r;
    bool all_true, inside;

    for(j = 0; j < (2*d); ++j)
    {
        
        Nd = num_particles[j];
        N = (int) Nd;
        r = dist(rng);
        if (r < (Nd-N))
        {
            N += 1;
        }
        num = 0;
        std::vector<double> xs(N*dim), vs(N*dim);
        while (num < N)
        {
            n = N-num;
            std::vector<double> new_xs(dim*n);
            std::vector<double> new_vs(dim*n);
            new_xs = sample_positions(n, j);
            new_vs = mv->sample(n, j);
            for (i = 0; i < n; ++i)
            {
                r = dist(rng);
                inside = true;
                for (k = 0; k < dim; ++k)
                {
                    new_xs[i*dim + k] += dt*r*new_vs[i*dim + k];
                    if (new_xs[i*dim+k] < 0.0 || new_xs[i*dim+k] > Ld[k])
                    {
                        inside = false;
                        break;
                    }
                }
                if (inside)
                {
                    for (k = 0; k < dim; ++k)
                    {
                        xs[num*dim+k] = new_xs[i*dim+k];
                        vs[num*dim+k] = new_vs[i*dim+k];
                    }
                    num++;
                }
            }
            // for (i = 0; i < n; ++i)
            // {
            //     for (k = 0; k < dim; ++k)
            //     {
            //         if (new_xs[i*dim+k] >= 0.0 && new_xs[i*dim+k] <= Ld[k])
            //         {
            //             all_true = true;
            //             break;
            //         }else{
            //             all_true = false;
            //             break;
            //         }
            //     }
            //     if (all_true)
            //     {
            //         for (k = 0; k < dim; ++k)
            //         {
            //             xs[num*dim+k] = new_xs[i*dim+k];
            //             vs[num*dim+k] = new_vs[i*dim+k];
            //         }
            //         num++;
            //     }
            // }
        }
        // ofstream file;
        // file.open("v_" + to_string(j) + ".txt");
        // for (const auto &e : vs)
        //     file << e << "\n";
        // file.close();
        // ofstream file1;
        // file1.open("x_" + to_string(j) + ".txt");
        // for (const auto &e : xs)
        //     file1 << e << "\n";
        // file1.close();
        pop.add_particles(xs, vs);
    }   
}

std::vector<double> Injector::sample_positions(int n, int j)
{
    std::vector<double> xs(n*dim);
    int i, k;
    for (i = 0; i < n; ++i)
    {
        xs[index[j%d]+i*dim] = L[j];
        for (k = 0; k < dim - 1; ++k)
        {
            xs[slices[(j%d)*(dim-1)+k]+i*dim] = dists[slices[(j%d)*(dim-1)+k]](rng);
        }
    }
    return xs;
}

void Injector::initialize_injection()
{
    this->d = mv->dim_nonperiodic;
    std::vector<double> L(2*d);
    std::vector<int> index(d);
    std::vector<int> slices(d*(dim-1));
    std::vector<double> surface_area(d);
    std::vector<std::uniform_real_distribution<double>> dists(dim);

    double area = 1.0;

    for(auto &l: Ld)
    {
        area *= l;
    }

    int i, k, m, j = 0;
    for(i = 0; i < dim; ++i)
    {
        dists[i] = distribution(0.0, Ld[i]);
        if(!periodic[i])
        {
            L[j] = 0.0;
            L[j+d] = Ld[i];
            surface_area[j] = area/Ld[i];
            index[j] = i;
            m = 0;
            for (k = (i+1); k < (i+dim); k++)
            {
                slices[j*(dim-1)+m] = k%dim;
                m++;
            }
            j++;
        }
    }

    std::vector<double> num_particles = mv->flux_number(plasma_density, 
                                                   surface_area, dt);
    std::cout <<"------Injector-------"<<'\n';
    std::cout << "surface_area: " << plasma_density << '\n';
    for (auto &ss : surface_area)
    {
        std::cout << ss << "  ";
    }
    std::cout << '\n';
    std::cout << "num_particles" << '\n';
    for (auto &ss : num_particles)
    {
        std::cout << ss << "  ";
    }
    std::cout << '\n';
    this->num_particles = num_particles;
    this->slices = slices;
    this->index = index;
    this->L = L;
    this->surface_area = surface_area;
    this->dists = dists;
}