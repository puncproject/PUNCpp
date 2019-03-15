#include <cstddef>
#include <iostream>
#include <vector>
#include <chrono>
#include <iterator>
#include <algorithm>
#include <iomanip>

using namespace std;

class Timer
{
  public:
    /**
       * Timer constructor
       * Starts the timing.
       */
    Timer(std::vector<std::string> tasks, std::size_t steps) : tasks(tasks),
                                                               steps(steps), 
                                                               times(std::vector<double>(tasks.size(),0.0)),
                                                               _begin(_clock::now()) {}
    /**
       * Resets the timer to current time
       */
    void reset() { _begin = _clock::now(); }
    /**
       * Calculates the elapsed time
       * @return The time elapsed
       */
    void progress(std::size_t n)
    {
        auto time_taken = std::chrono::duration_cast<_second>(_clock::now() - _begin).count();
        auto percent = (double) n / steps;
        auto time_left = time_taken * (1.0 / percent - 1.0);

        std::cout << "Step " << n << " of " << steps;
        std::cout<<". Time remaining: ";
        formatter(time_left);
        std::cout<<'\n';
    }

    void tic(std::string tag)
    {
        _index = std::distance(tasks.begin(), std::find(tasks.begin(), tasks.end(), tag));
        _time = _clock::now();
    }
    
    void toc()
    {
        times[_index] += std::chrono::duration_cast<_second>(_clock::now() - _time).count();
    }

    double elapsed() const
    {
        return std::chrono::duration_cast<_second>(_clock::now() - _begin).count();
    }

    void summary()
    {
        auto total_time = elapsed();

        int len=0;
        for(int i = 0; i<times.size(); ++i)
        {
            len = tasks[i].length() > len ? tasks[i].length() : len;
        }
        std::vector<int> blanks(times.size());
        for (int i = 0; i < times.size(); ++i)
        {
            blanks[i] = len - tasks[i].length() + 12;
        }
        std::cout<<"-----------------------------------------------------------"<<'\n';
        std::cout<<"                      Summary of tasks                     "<<'\n';
        std::cout<<"-----------------------------------------------------------"<<'\n';
        std::cout<<" Task               Time             Percentage          "<<'\n'; 
        std::cout<<"-----------------------------------------------------------"<<'\n';
        for(int i = 0; i<times.size(); ++i)
        {
            std::cout << tasks[i] << std::setw(blanks[i]) << " ";
            formatter(times[i]);
            std::cout << "            " << 100 * times[i] / total_time << '\n';
        }
        
        std::cout<<"-----------------------------------------------------------"<<'\n';
        std::cout << "            Total run time:    ";
        formatter(total_time);
        std::cout << '\n';
        std::cout<<"-----------------------------------------------------------"<<'\n';
    }

    void formatter(double time_range)
    {
        auto duration = std::chrono::duration<double>(time_range);
        days day = std::chrono::duration_cast<days>(duration);
        duration -= day;
        std::chrono::hours hour = std::chrono::duration_cast<std::chrono::hours>(duration);
        duration -= hour;
        std::chrono::minutes min = std::chrono::duration_cast<std::chrono::minutes>(duration);
        duration -= min;
        std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>(duration);
        duration -= sec;

        std::cout << day.count() << " d " << hour.count() << ':'
                  << min.count() << ':' << sec.count();
    }

  private:
    std::vector<std::string> tasks;
    std::vector<double> times;
    std::size_t steps;

    typedef std::chrono::high_resolution_clock _clock;
    typedef std::chrono::duration<double, std::ratio<1>> _second;
    typedef std::chrono::duration<int, std::ratio_multiply<std::chrono::hours::period, std::ratio<24>>::type> days;
 
    std::chrono::time_point<_clock> _begin;
    std::chrono::time_point<_clock> _time;
    int _index;
};

void fib()
{
    int t1 = 0, t2 = 1, nextTerm = 0, n=100000;

    nextTerm = t1 + t2;

    while (nextTerm <= n)
    {
        t1 = t2;
        t2 = nextTerm;
        nextTerm = t1 + t2;
    }
}

void vec()
{
    std::vector<int> v(1000000,0);
    for(int i = 0; i<1000000;++i)
    {
        v[i] = i;
    }
}

int main()
{
    std::size_t steps = 100;
    std::vector<std::string> tasks{"fib", "vector"};

    Timer timer(tasks, steps);

    for(int i = 0; i<steps;++i)
    {
        timer.tic(tasks[0]);
        fib();
        timer.toc();
        timer.tic(tasks[1]);
        vec();
        timer.toc();

        timer.progress(i);
    }
    timer.summary();

    return 0;
}