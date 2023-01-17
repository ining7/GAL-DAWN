#ifndef TIMER_HPP
#define TIMER_HPP
#include <chrono>
#include <iostream>
#include <string>

class Timer {
 public:
  Timer() = delete;

  Timer(const std::string& name) : name(name), usingTime(0), times(0) {
    beginTime = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }

  ~Timer() {
    if (times == 0) {
      auto endTime = static_cast<double>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());
      usingTime += endTime - beginTime;

      std::cout << name << " AllTime: " << usingTime << " ms" << std::endl;
    } else {
      std::cout << name << " AllTime: " << usingTime << " ms" << std::endl;
    }
  }
  void begin() {
    beginTime = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }
  void end() {
    times++;
    auto endTime = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    usingTime += endTime - beginTime;
  }

  double getAvgTime() { return usingTime / (times + 0.0); }

  double getAllTime() { return usingTime; }

 private:
  std::string name;
  int times;
  double beginTime;
  double usingTime;
};

/* Example:

int a() { }
int b() { }
int c() { }

int main(){
    Timer bTimer("b");
    for (size_t i = 0; i < 100; ++i) {
        a();
        bTimer.begin();
        b();
        bTimer.end();
        c();
    }
    return 0;
}
*/

#endif  // TIMER_HPP
