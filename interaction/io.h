// Copyright (C) 2018, Diako Darian and Sigvald Marholm
//
// This file is part of PUNC++.
//
// PUNC++ is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// PUNC++ is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// PUNC++. If not, see <http://www.gnu.org/licenses/>.

/**
 * @file		parser.h
 * @brief		Helper functions for parsing options from ini file
 */

#ifndef IO_H
#define IO_H

#include <punc/population.h>

#include <string>
#include <vector>
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace punc {

bool split_suffix(const string &in, string &body, string &suffix, const vector<string> &suffixes);

/**
 * @brief Splits a string by spaces to a vector of type T
 * @param   str     String to split
 * @return          Vector
 */
template <typename T>
vector<T> str_to_vec(const string &str)
{
    std::istringstream iss(str);
    vector<T> result;
    T tmp;
    while(iss >> tmp) result.push_back(tmp);
    return result;
}


/**
 * @brief Option parser
 * 
 * This class extends the functionality of boost::program_options according
 * to our needs. Specifically we need to be able to read repeated options,
 * vector valued quantities, and quantities with suffixes. Consider for
 * instance the following excerpt from an input file:
 *
 * @code
 *  mesh = my_mesh.xml
 *  dt = 0.2
 *  B = 0.1 0 0
 *
 *  [species]
 *  charge = -1
 *  vdrift = 1 0 0
 *  amount = 10 per cell
 *
 *  [species]
 *  vdrift = 1 0 0
 *  amount = 10 per cell
 *
 *  [species]
 *  charge = 3
 *  vdrift = 0 0 0
 *  amount = 10 per cell
 *  
 * @endcode
 *
 * - dt is a double, mesh is a string.
 * - dt is a scalar, B is a vector.
 * - dt is a single value, but charge is repeated multiple times.
 * - dt has no suffix, but amount is specified per cell.
 *
 * In addition, it is often desirable to require a certain length of a vector
 * or number of repetitions. For instance, in the above example charge is
 * by mistake specified only twice, which yields a vector of length two.
 * Below follows examples of how to obtain the values in mesh, B and amount:
 *
 * @code
 *  Options opt(vm);
 *
 *  string mesh;
 *  opt.get("mesh", mesh);
 *
 *  size_t dims = 3;
 *  vector<double> B(dims); // Zero vector as default
 *  opt.get_vector("B", B, dims, true);
 *
 *  size_t n_species = 2;
 *  vector<double> amount;
 *  string amount_suffix;
 *  opt.get_repeated("species.amount", amount, n_species,
 *                   {"per cell", "per volume"}, amount_suffix);
 * @endcode
 *
 * Beware that parsing the options has a lot of overhead, and should not be
 * done in time critical parts of the program (e.g. the time loop).
 */
class Options {
public:
    Options(po::variables_map vm) : vm(vm) {};
    
    /**
     * @brief Get an entry with a suffix
     * @param       key         Key for which to get the value and suffix.
     * @param[out]  res         The value of the entry.
     * @param       optional    Whether the key is optional or mandatory.
     *
     * If an optional key is not present, the value already in res will
     * be left untouched, and acts as a default value.
     */
    template <typename T>
    void get(const string &key, T &res, bool optional = false) const {
        string suffix; // throw away
        get(key, res, {""}, suffix, optional);
    };

    /**
     * @brief Get an entry with a suffix
     * @param       key         Key for which to get the value and suffix.
     * @param[out]  res         The value of the entry.
     * @param       suffixes    Vector of valid suffixes.
     * @param[out]  suffix      The suffix of the entry.
     * @param       optional    Whether the key is optional or mandatory.
     *
     * To make suffixes optional, include the empty string "" in suffixes.
     * Beware that this will always trigger a match, and the search for 
     * a suffix stops on the first match, so this empty string must be the
     * last element.
     *
     * If an optional key is not present, the value already in res will
     * be left untouched, and acts as a default value.
     */
    template <typename T>
    void get(const string &key, T &res, const vector<string> &suffixes,
             string &suffix, bool optional = false) const {

        vector<T> res_;
        get_vector(key, res_, 1, suffixes, suffix, optional);
        if(res_.size()){ // Empty for non-present, optional options
            res = res_[0];
        }
    };

    /**
     * @brief Get repeated entries
     * @param       key         Key for which to get values and suffixes.
     * @param[out]  res         Vector of values for each entry.
     * @param       num         Number of entries. 0 for arbitrary number.
     * @param       optional    Whether the key is optional or mandatory.
     *
     * If an optional key is not present, the value already in res will
     * be left untouched, and acts as a default value.
     */
    template <typename T>
    void get_repeated(const string &key, vector<T> &res, size_t num,
                      bool optional = false) const {

        vector<string> suffix; // throw away
        get_repeated(key, res, num, {""}, suffix, optional);
    };

    /**
     * @brief Get repeated entries with suffixes
     * @param       key         Key for which to get values and suffixes.
     * @param[out]  res         Vector of values for each entry.
     * @param       num         Number of entries. 0 for arbitrary number.
     * @param       suffixes    Vector of valid suffixes.
     * @param[out]  suffix      Vector of suffixes for each entry.
     * @param       optional    Whether the key is optional or mandatory.
     *
     * To make suffixes optional, include the empty string "" in suffixes.
     * Beware that this will always trigger a match, and the search for 
     * a suffix stops on the first match, so this empty string must be the
     * last element.
     *
     * If an optional key is not present, the value already in res will
     * be left untouched, and acts as a default value.
     */
    template <typename T>
    void get_repeated(const string &key, vector<T> &res, size_t num,
                      const vector<string> &suffixes, vector<string> &suffix,
                      bool optional = false) const {

        vector<vector<T>> res_;
        get_repeated_vector(key, res_, 1, num, suffixes, suffix, optional);
        if(res_.size()){ // Empty for non-present, optional options
            res = vector<T>();
            for(auto &r : res_) res.push_back(r[0]);
        }
    };


    /**
     * @brief Get a vector valued entry
     * @param       key         Key for which to get a value and suffix.
     * @param[out]  res         The vector value of the entry.
     * @param       len         Length of the vector. 0 for arbitrary length.
     * @param       optional    Whether the key is optional or mandatory.
     *
     * If an optional key is not present, the value already in res will
     * be left untouched, and acts as a default value.
     */
    template <typename T>
    void get_vector(const string &key, vector<T> &res, size_t len,
                    bool optional = false) const {

        string suffix; // throw away
        get_vector(key, res, len, {""}, suffix, optional);
    }

    /**
     * @brief Get a vector valued entry with a suffix
     * @param       key         Key for which to get a value and suffix.
     * @param[out]  res         The vector value of the entry.
     * @param       len         Length of the vector. 0 for arbitrary length.
     * @param       suffixes    Vector of valid suffixes.
     * @param[out]  suffix      The suffix of the entry.
     * @param       optional    Whether the key is optional or mandatory.
     *
     * To make suffixes optional, include the empty string "" in suffixes.
     * Beware that this will always trigger a match, and the search for 
     * a suffix stops on the first match, so this empty string must be the
     * last element.
     *
     * If an optional key is not present, the value already in res will
     * be left untouched, and acts as a default value.
     */
    template <typename T>
    void get_vector(const string &key, vector<T> &res, size_t len,
                    const vector<string> &suffixes,
                    string &suffix,
                    bool optional = false) const {

        vector<vector<T>> res_;
        vector<string> suffix_;
        get_repeated_vector(key, res_, len, 1, suffixes, suffix_, optional);
        if(res_.size()){ // Empty for non-present, optional options
            res = res_[0];
            suffix = suffix_[0];
        }

    }

    /**
     * @brief Get repeated vector valued entries
     * @param       key         Key for which to get values and suffixes.
     * @param[out]  res         Vector of vector values for each entry.
     * @param       len         Length of vectors. 0 for arbitrary length.
     * @param       num         Number of entries. 0 for arbitrary number.
     * @param       optional    Whether the key is optional or mandatory.
     *
     * If an optional key is not present, the value already in res will
     * be left untouched, and acts as a default value.
     */
    template <typename T>
    void get_repeated_vector(const string &key, vector<vector<T>> &res,
                             size_t len, size_t num,
                             bool optional = false) const {

        vector<string> suffix; // throw away
        get_repeated_vector(key, res, len, num, {""}, suffix, optional);
    }

    /**
     * @brief Get repeated vector valued entries with suffixes
     * @param       key         Key for which to get values and suffixes.
     * @param[out]  res         Vector of vector values for each entry.
     * @param       len         Length of vectors. 0 for arbitrary length.
     * @param       num         Number of entries. 0 for arbitrary number.
     * @param       suffixes    Vector of valid suffixes.
     * @param[out]  suffix      Vector of suffixes for each entry.
     * @param       optional    Whether the key is optional or mandatory.
     *
     * To make suffixes optional, include the empty string "" in suffixes.
     * Beware that this will always trigger a match, and the search for 
     * a suffix stops on the first match, so this empty string must be the
     * last element.
     *
     * If an optional key is not present, the value already in res will
     * be left untouched, and acts as a default value.
     */
    template <typename T>
    void get_repeated_vector(const string &key, vector<vector<T>> &res,
                             size_t len, size_t num,
                             const vector<string> &suffixes,
                             vector<string> &suffix,
                             bool optional = false) const {

        if(vm.count(key)){

            vector<string> bodies = vm[key].as<vector<string>>();
            if(num != 0 && bodies.size() != num){
                cerr << "Expected " << num << " " << key << " parameters" << endl;
                exit(1);
            }

            res = vector<vector<T>>();
            suffix = vector<string>();

            for(auto &body : bodies){
                string head, tail;
                bool found = split_suffix(body, head, tail, suffixes);
                if(!found){
                    cerr << key << " has invalid suffix. ";
                    cerr << "Valid suffixes:";
                    for(auto &s : suffixes) cerr << " \"" << s << "\"";
                    cerr << endl;
                    exit(1);
                }
                vector<T> tmp = str_to_vec<T>(head);
                if(len != 0 && tmp.size() != len){
                    cerr << "Expected " << key << " to be vector of ";
                    cerr << "length " << len << endl;
                    exit(1);
                }
                res.push_back(tmp);
                suffix.push_back(tail);
            }

        } else if (!optional){

            cerr << "Missing mandatory parameter " << key << endl;
            exit(1);
        }
    }

private:
    po::variables_map vm;
};

/**
 * @brief Reads all species from options
 * @param   options     Options from ini-file and CLI
 * @param   mesh        The mesh
 * @return              Vector of all species
 */
vector<Species> read_species(const Options &opt, const Mesh &mesh);

} // namespace punc

#endif // IO_h
