#include "punc/injector.h"
#include "punc/population.h"
#include <chrono>
#include <ctime>

using namespace std;

auto pdf(vector<double> Ld)
{
    double A = 0.5, mode = 1.0;
    auto f = [A, mode, Ld](vector<double> x)->double{
        return 1.0 + A*sin(mode*2*M_PI*x[0]/Ld[0]);
    };
    return f;
}

auto pdf_flux(double d, int s)
{
     auto pdf = [d, s](double& t)->double{
                return s*t*exp(-.5*(t/(s*(1.-t))-d)*(t/(s*(1.-t))-d))/\
                    ((s*(1.-t))*(s*(1.-t))*(s*(1.-t)));};
     return pdf;
}
double pdf_max_drifting(double vd_n, int s)
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
double pdf_max_drifting_maxwellian(function<double (double&)> pdf, double d, int s)
{
    double ca = 2.0;
    double cb = -4.0 - s*d;
    double cc = s*d;
    double cd = 1.0;

    double p = (3.0*ca*cc - cb*cb)/(3.0*ca*ca);
    double q = (2.0*cb*cb*cb - 9.0*ca*cb*cc + 27*ca*ca*cd) / (27*ca*ca*ca);

    double sqrt_term = sqrt(-p/3.0);
    double root = 2.0*sqrt_term*\
           cos(acos(3.0*q/(2.0*p*sqrt_term))/3.0 - 2.0*M_PI/3.0);
    double tmp = root - cb/(3.0*ca);
    return pdf(tmp);
}

void test_srs_1D(const int N)
{
    vector<double> vs(N);
    double v_thermal = 1.0, d = 1.0;
    int s = 1;
    auto pdf = pdf_flux(d, s);
    auto transform = [v_thermal](double& t){ return v_thermal*t/(1.0-t);};
    double pdf_max = pdf_max_drifting_maxwellian(pdf,d,s);

    SRS rs(pdf, pdf_max, transform);

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();
    vs = rs.sample(N);
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms dd = std::chrono::duration_cast<ms>(fs);
    std::cout << fs.count() << "s\n";
    std::cout << dd.count() << "ms\n";

    // ofstream out;
    // out.open("v.txt");
    // for (const auto &e : vs) out << e << "\n";
}
void test_srs_ND(const int N)
{
    vector<double> vs(N);
    vector<double> Ld = {2*M_PI, 2*M_PI};
    double A = 0.5;

    auto f = pdf(Ld);
    double pdf_max = A+1.0;

    SRS rs(f, pdf_max, Ld);

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();
    vs = rs.sample(N);
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms dd = std::chrono::duration_cast<ms>(fs);
    std::cout << fs.count() << "s\n";
    std::cout << dd.count() << "ms\n";

    // ofstream out;
    // out.open("v.txt");
    // for (const auto &e : vs) out << e << "\n";

}

void test_ors(const int N)
{
    double vd_n = 6.0;
    double vth = 1.0;
    double si = -1.0;

    double ro = pdf_max_drifting(vd_n, si);

    vector<double> vs(N);

    vector<double> root;
    root.push_back(pdf_max_drifting(vd_n, si));

    auto V = [vd_n, si](double& t){ return -log(t) + \
    3.*log(1.-t) + 0.5*(vd_n - si*t/(1.-t))*(vd_n - si*t/(1.-t));};
    
    auto dV = [vd_n, si](double& t){ return (-1./t) - \
    (3./(1.-t)) + (si*(t*(si+vd_n)-vd_n)/((1.-t)*(1.-t)*(1.-t)));};
    
    auto transform = [vth, si](double& t){return si*vth*t/(1.-t);};


    ORS rs(V, dV, root, transform);

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();
    vs = rs.sample(N);
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms dd = std::chrono::duration_cast<ms>(fs);
    std::cout << fs.count() << "s\n";
    std::cout << dd.count() << "ms\n";

    ofstream out;
    out.open("v.txt");
    for (const auto &e : vs) out << e << "\n";
}

void test_maxwellian(int N)
{
    double vth{1.0};

    vector<double> vd{0.,0.};
    vector<bool> periodic{false, false};

    int dim = vd.size();
    Maxwellian mv(vd, vth, periodic);
    int dim_nonperiodic = 0;
    for(auto &&p: periodic)
    {
        if(!p)
        {
            dim_nonperiodic++;
        }
    }

    int len = dim*dim_nonperiodic;
    int dd = dim*dim;
    vector<function<double (double&)>> pdf(2*dd);
    vector<function<double (double&)>> pdfs(2*len);
    vector<double> s{1.0, -1.0};


    vector<int> v_n(dd, 0);
    int i, k, j = 0;
    for (i = 0; i < dim; ++i)
    {
        v_n[j+ i*dim] = 1;
        j++;
    }

    
    for(i = 0; i<dim; ++i)
    {
        for (j = 0; j<dim; ++j)
        {
            k = i*dim +j;
            if (vd[j] != 0)
            {
                if (v_n[k]==1)
                {
                    pdf[k] = mv.pdf_flux_drifting(k%dim, s[0]);
                    pdf[k+dd] = mv.pdf_flux_drifting(k%dim, s[1]);
                }else{
                    pdf[k] = mv.pdf_maxwellian(k%dim);
                    pdf[k+dd] = mv.pdf_maxwellian(k%dim);                    
                }
            }else{
                if(v_n[k]==1)
                {
                    pdf[k] = mv.pdf_flux_nondrifting(s[0]);
                    pdf[k+dd] = mv.pdf_flux_nondrifting(s[1]);
                }else{
                    pdf[k] = mv.pdf_maxwellian(k%dim);
                    pdf[k+dd] = mv.pdf_maxwellian(k%dim);
                }
            }
        }
    }

    int num = 500;
    double lb = -10.0, ub = 10.0;
    vector<double> vs, xs, fx(dim*num);
    
    xs = linspace(lb,ub,num);
    k = 0;
    for (int i=0; i<dim; i++)
    {
        if(!periodic[i])
        {
            for(int j=(i*dim); j<((i+1)*dim); j++)
            {
                pdfs[k] = pdf[j];
                pdfs[k+len] = pdf[j+dd];
                k++;
            }
        }
    }

    for (int ii = 0; ii<(2*dim_nonperiodic); ++ii)
    {
        for(int i=0; i<dim; i++)
        {
            for(int j=0; j<num; j++)
            {
                fx[i*num+j] = pdfs[ii*dim+i](xs[j]);
            }
        }
        vs = mv.sample(N,ii);

        ofstream file1;
        file1.open ("v_"+to_string(ii) + ".txt"); 
        for (const auto &e : vs) file1 << e << "\n";
        file1.close();

        ofstream file2;
        file2.open ("f_"+to_string(ii) + ".txt"); 
        for (const auto &e : fx) file2 << e << "\n";
        file2.close();
    }
}

void test_injector(int N)
{

    double dt{0.1}, vth{1.0};
    vector<double> Ld{2., 4.,6.}, vd{0.,0.,0.};
    vector<bool> periodic{false, false, false};

    Population pop(N, Ld, periodic, vth, vd);
    Injector inj(pop, dt);

    cout << "surface_area: "<<inj.plasma_density<<'\n';
    for(auto & ss: inj.surface_area)
    {
        cout <<ss<<"  ";
    }
    cout <<'\n';
    cout << "num_particles" << '\n';
    for (auto &ss : inj.num_particles)
    {
        cout << ss << "  ";
    }
    cout << '\n';
    cout << "L" << '\n';
    for (auto &ss : inj.L)
    {
        cout << ss << "  ";
    }
    cout << '\n';
    cout << "index" << '\n';
    for (auto &ss : inj.index)
    {
        cout << ss << "  ";
    }
    cout << '\n';
    cout << "slices" << '\n';
    for (auto &ss : inj.slices)
    {
        cout << ss << "  ";
    }
    cout << '\n';
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();
    inj.inject();
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms dd = std::chrono::duration_cast<ms>(fs);
    std::cout << fs.count() << "s\n";
    std::cout << dd.count() << "ms\n";
}

void test_population(int N)
{
    double dt{0.1}, vth{1.0};
    vector<double> Ld{2.,2.}, vd{0.,0.};
    vector<bool> periodic{false, false};

    Population pop(N, Ld, periodic, vth, vd);
    Injector inj(pop, dt);
    Move mv(pop, periodic, dt);

    int i, itr = 10000;
    int dim = Ld.size();
    vector<int> count, out;
    count.push_back(N);

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0 = Time::now();
    for (i=0; i<itr; ++i)
    {
        cout <<"iteration: "<<i<<'\n';
        mv.move();
        out.push_back(pop.tot_num);
        inj.inject();
        count.push_back(pop.tot_num);
        cout << "tot_num: "<< pop.tot_num<<'\n';
    }
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms dd = std::chrono::duration_cast<ms>(fs);
    std::cout << fs.count() << "s\n";
    std::cout << dd.count() << "ms\n";    

    ofstream file, file1;
    file.open("c.txt");
    for (const auto &e : count)
        file << e << "\n";
    file.close();
    file1.open("o.txt");
    for (const auto &e : out)
        file1 << e << "\n";
    file1.close();
}

int main()
{
    int N = 10000000;
    // test_ors(N);
    // test_srs_1D(N);
    // test_srs_ND(N);
    // test_maxwellian(N);
    // test_injector(N);
    test_population(N);

    
    return 0;
}
