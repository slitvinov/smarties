//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_TaskQueue_h
#define smarties_TaskQueue_h

#include <functional>
#include <utility>
#include <vector>

namespace smarties
{

class TaskQueue
{
  using cond_t = std::function<bool()>;
  using func_t = std::function<void()>;
  std::vector<std::pair<cond_t, func_t>> tasks;

public:
  inline void add(cond_t && cond, func_t && func) {
    tasks.emplace_back(std::move(cond), std::move(func));
  }

  inline void run()
  {
    // go through task list once and execute all that are ready:
    for(size_t i=0; i<tasks.size(); ++i) if(tasks[i].first()) tasks[i].second();
  }
};

} // end namespace smarties
#endif // smarties_Settings_h
