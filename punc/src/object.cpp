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
 * @file		object.cpp
 * @brief		Implementing objects
 */

#include "../include/punc/object.h"
#include <vector>

namespace punc
{

/*******************************************************************************
 * LOCAL DECLARATIONS
 ******************************************************************************/

/**
 * @brief Identifies the set of objects sharing the charge with a given object
 * @param       vsources    List of voltage sources
 * @param       node        Object to look for
 * @param[out]  group       List of objects sharing charge with node
 */
static void get_charge_sharing_set(SourceVector &vsources, int node, std::vector<int> &group);

/**
 * @brief Identifies all sets of charge sharing objects
 * @param   vsources        List of voltage sources
 * @param   num_objects     Number of objects
 * @return                  List of charge sharing sets
 *
 * Objects connected through voltage sources share the charge, and unless they
 * are connected with a fixed voltage with respect to ground they have a common
 * charge constraint. This follows the voltage sources to identify which objects
 * must have a charge constraint.
 */
static std::vector<std::vector<int>> get_charge_sharing_sets(SourceVector vsources, int num_objects);

/*******************************************************************************
 * GLOBAL DEFINITIONS
 ******************************************************************************/

Circuit::Circuit(const ObjectVector &object_vector,
                 const SourceVector &vsources,
                 const SourceVector &isources)
                :vsources(vsources), isources(isources){

    num_objects = object_vector.size();
    groups = get_charge_sharing_sets(vsources, num_objects);

}

// std::ostream& operator<<(std::ostream& out, const VSource &s){
//     return cout << "V_{" << s.node_a << ", " << s.node.b << "} = " << s.value;
// }
// 
// std::ostream& operator<<(std::ostream& out, const ISource &s){
//     return cout << "I_{" << s.node_a << ", " << s.node.b << "} = " << s.value;
// }

/*******************************************************************************
 * LOCAL DEFINITIONS
 ******************************************************************************/

static void get_charge_sharing_set(SourceVector &vsources, int node, std::vector<int> &group)
{
    group.emplace_back(node);

    std::size_t i = 0;
    while (i < vsources.size())
    {
        Source vsource = vsources[i];
        if (vsource.node_a == node)
        {
            vsources.erase(vsources.begin() + i);
            get_charge_sharing_set(vsources, vsource.node_b, group);
        }
        else if (vsource.node_b == node)
        {
            vsources.erase(vsources.begin() + i);
            get_charge_sharing_set(vsources, vsource.node_a, group);
        }
        else
        {
            i += 1;
        }
    }
}

static std::vector<std::vector<int>> get_charge_sharing_sets(SourceVector vsources, int num_objects)
{
    std::vector<int> nodes(num_objects);
    std::iota(std::begin(nodes), std::end(nodes), 0);

    std::vector<std::vector<int>> groups;

    while (vsources.size() != 0)
    {
        std::vector<int> group;
        get_charge_sharing_set(vsources, vsources[0].node_a, group);
        groups.emplace_back(group);
    }
 
    for (std::size_t i = 0; i != groups.size(); i++){
        for (std::size_t j = 0; j != groups[i].size(); j++){
            if (groups[i][j] != -1){
                nodes.erase(std::remove(nodes.begin(), nodes.end(), groups[i][j]), nodes.end());
            }
        }
    }

    groups.erase(std::remove_if(groups.begin(), groups.end(), [](std::vector<int> a) { return std::find(a.begin(), a.end(), -1) != a.end(); }), groups.end());

    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
        std::vector<int> node{nodes[i]};
        groups.emplace_back(node);
    }
    return groups;
}

} // namespace punc
