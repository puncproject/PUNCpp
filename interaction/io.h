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

#ifndef PARSER_H
#define PARSER_H

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
 * @brief Safely get repeated options
 * @param   key     Key to the options
 * @param   num     Number of options that should be present
 * @param   def     Default value in case no option is present
 * @see get_vector, get_repeated_vector
 *
 * This is when you want to make sure that an option is repeated exactly num
 * times, or not at all (in which case it defaults to def). For instance:
 *
 * @code
 *      [species]
 *      mass = 1
 *
 *      [species]
 *      mass = 1836
 * @endcode
 *
 * As a counter example, if there were three species, and the middle one lacked
 * a mass, reading the array using options[key].as<vector<T>>() would return an
 * array of only two masses. This function handles this safely.
 */
template <typename T>
vector<T> get_repeated(po::variables_map options, string key, size_t num, T def){
    vector<T> res;
    if(options.count(key)){
        res = options[key].as<vector<T>>();
        if(res.size() != num){
            cerr << "Wrong number of " << key << " specified" << endl;
            exit(1);
        }
    } else {
        res = vector<T>(num, def);
    }
    return res;
}


/**
 * @brief Safely get vector option
 * @param   key     Key to the options
 * @param   len     Length the vector should have
 * @param   def     Default vector in case no option is present
 *
 * This is for a single vector-valued option like this:
 *
 * @code
 *      B = 1.0 0.0 0.0
 * @endcode
 *
 * It checks that the vector has the correct amount of parameters.
 */
template <typename T>
vector<T> get_vector(po::variables_map options,
                     string key, size_t len, vector<T> def)
{
    if(options.count(key)){
        string str = options[key].as<string>();
        vector<T> res = str_to_vec<T>(str);
        if(res.size() != len){
            cerr << "Wrong number of elements in " << key << endl;
            exit(1);
        }
        return res;
    } else {
        return def;
    }
}

/**
 * @brief Safely get repeated vector options
 * @param   key     Key to the options
 * @param   num     Numbe of options that should be present
 * @param   len     Length the vector should have
 * @param   def     Default vector in case no option is present
 * @see get_repeated, get_vector
 *
 * This is for repeated vector-valued options like this:
 *
 * @code
 *      [species]
 *      vdrift = 1.0 0.0 0.0
 *
 *      [species]
 *      vdrift = 1.0 0.0 0.0
 * @endcode
 *
 * It ensures the correct number of repetions, as well as the correct number
 * of options.
 */
template <typename T>
vector<vector<T>> get_repeated_vector(po::variables_map options,
        string key, size_t num, size_t len, vector<T> def){

    vector<vector<T>> res;
    if(options.count(key)){
        vector<string> str_vec = options[key].as<vector<string>>();
        if(str_vec.size() != num){
            cerr << "Wrong number of " << key << " specified" << endl;
            exit(1);
        }
        for(auto &str : str_vec){
            vector<T> tmp = str_to_vec<T>(str);
            if(tmp.size() != len){
                cerr << "Wrong number of elements in " << key << endl;
                exit(1);
            }
            res.push_back(tmp);
        }
    } else {
        for(size_t i=0; i<num; i++){
            res.push_back(def);
        }
    }
    return res;
}

/**
 * @brief Reads all species from options
 * @param   options     Options from ini-file and CLI
 * @param   mesh        The mesh
 * @return              Vector of all species
 */
vector<Species> read_species(po::variables_map options, const Mesh &mesh);

} // namespace punc

#endif // PARSER_H
