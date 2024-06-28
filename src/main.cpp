// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: Fri Jan 22 20:45:07 2021
//      Author: Kiwon Um
//        Mail: kiwon.um@telecom-paris.fr
//
// Description: SPH simulator (DO NOT DISTRIBUTE!)
//
// Copyright 2021-2024 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um,
// Telecom Paris, France. The program(s) may be used and/or copied only with
// the written permission of Kiwon Um or in accordance with the terms and
// conditions stipulated in the agreement/contract under which the program(s)
// have been supplied.
// ----------------------------------------------------------------------------

#define _USE_MATH_DEFINES

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.141592
#endif

#include "Vector.hpp"

// window parameters
GLFWwindow *gWindow = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;

// timer
float gAppTimer = 0.0;
float gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gShowGrid = true;
bool gShowVel = false;
int gSavedCnt = 0;
const int kViewScale = 15;
int iterations = 0;
float delta;



// SPH Kernel function: cubic spline
class CubicSpline {
public:
  explicit CubicSpline(const Real h=1) : _dim(2)
  {
    setSmoothingLen(h);
  }
  void setSmoothingLen(const Real h)
  {
    const Real h2 = square(h), h3 = h2*h;
    _h = h;
    _sr = 2e0*h;
    _c[0]  = 2e0/(3e0*h);
    _c[1]  = 10e0/(7e0*M_PI*h2);
    _c[2]  = 1e0/(M_PI*h3);
    _gc[0] = _c[0]/h;
    _gc[1] = _c[1]/h;
    _gc[2] = _c[2]/h;
  }
  Real smoothingLen() const { return _h; }
  Real supportRadius() const { return _sr; }

  Real f(const Real l) const
  {
    const Real q = l/_h;
    if(q<1e0) return _c[_dim-1]*(1e0 - 1.5*square(q) + 0.75*cube(q));
    else if(q<2e0) return _c[_dim-1]*(0.25*cube(2e0-q));
    return 0;
  }
  Real derivative_f(const Real l) const
  {
    const Real q = l/_h;
    if(q<=1e0) return _gc[_dim-1]*(-3e0*q+2.25*square(q));
    else if(q<2e0) return -_gc[_dim-1]*0.75*square(2e0-q);
    return 0;
  }

  Real w(const Vec2f &rij) const { return f(rij.length()); }
  Vec2f grad_w(const Vec2f &rij) const { return grad_w(rij, rij.length()); }
  Vec2f grad_w(const Vec2f &rij, const Real len) const
  {
    return derivative_f(len)*rij/len;
  }

private:
  unsigned int _dim;
  Real _h, _sr, _c[3], _gc[3];
};

class SphSolver {
public:
  explicit SphSolver(
    const Real nu=0.08, const Real h=0.5, const Real density=1e3,
    const Vec2f g=Vec2f(0, -9.8), const Real eta=0.01, const Real gamma=7.0) : //d0 init = 10e3
    _kernel(h), _nu(nu), _h(h), _d0(density),
    _g(g), _eta(eta), _gamma(gamma)
  {
    _dt = 0.0005;
    _m0 = _d0*_h*_h;
    _c = std::fabs(_g.y)/_eta;
    _k = _d0*_c*_c/_gamma;

    beta = _dt*_dt*_m0*_m0*2*((1/_d0)/_d0);
  }

  // assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
  // the size of f_width, f_height; each cell is sampled with 2x2 particles.
  void initScene(
    const int res_x, const int res_y, const int f_width, const int f_height)
  {
    _pos.clear();
    _pidxInGrid.resize(res_x * res_y);
    _resX = res_x;
    _resY = res_y;


    // set wall for boundary
    _l = 0.5*_h;
    _r = static_cast<Real>(res_x) - 0.5*_h;
    _b = 0.5*_h;
    _t = static_cast<Real>(res_y) - 0.5*_h;

    // sample a fluid mass
    for(int j=0; j<f_height; ++j) {
      for(int i=0; i<f_width; ++i) {
        _pos.push_back(Vec2f(i+0.25, j+0.25));
        _pos.push_back(Vec2f(i+0.75, j+0.25));
        _pos.push_back(Vec2f(i+0.25, j+0.75));
        _pos.push_back(Vec2f(i+0.75, j+0.75));
      }
    }



    _vel = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _predictedVel = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _predictedPos = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));

    _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _p   = std::vector<Real>(_pos.size(), 0);
    _Pa = std::vector<Real>(_pos.size(), 0);
    _d   = std::vector<Real>(_pos.size(), _d0); //d0 instead of 0 because we dont compute density first
    _predictedDensity = std::vector<Real>(_pos.size(), 0);
    _dErr   = std::vector<Real>(_pos.size(), 0);

    _col = std::vector<float>(_pos.size()*4, 1.0); // RGBA
    _vln = std::vector<float>(_pos.size()*4, 0.0); // GL_LINES

    updateColor();
  }


 float preComputeDelta(){
      int i = 400; //ok filled neighborhood 1.58443e+06
      float S1 = 0;
      float S2 = 0;

      for (tIndex j = 0; j < _pos.size(); j++) {
        S1 += _kernel.w(_pos[i] - _pos[j]);
        }

      for (tIndex j = 0; j < _pos.size(); j++) {
        S2 += (_kernel.w(_pos[i] - _pos[j])*_kernel.w(_pos[i] - _pos[j]));
        }

      return ((-1) / (beta*(-(S1*S1)-S2)));

 }



  void update()
  {
    std::cout << ".\n" << std::flush;

    //build neighborhood
    buildNeighbor();


    //compute accekeration
    //clear acc
    _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _acc_p = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0)); 

    //compute nonPressure forces
    applyBodyForce();
    applyViscousForce();

    initialization(); //set pressure to 0


    //pressure solver loop
    iterations = 0;
    while ((calculateAverage(_dErr) > (0.02 * _d0))
  || (iterations < 3)) { // while one of the conditions is true ||

        predictVelocity();
        predictPosition();
        resolveCollision();

        predictDensity();
        calculateDensityError();

        std::cout << "Density Error: " << calculateAverage(_dErr) << std::endl;
        
        calculatePressure();
        applyPressureForce();

        
        iterations++;
    }

    updateVelocity();
    updatePosition();

    resolveCollision();
    updateColor();

    if(gShowVel) updateVelLine();
  }

  tIndex particleCount() const { return _pos.size(); }
  const Vec2f& position(const tIndex i) const { return _pos[i]; }
  const float& color(const tIndex i) const { return _col[i]; }
  const float& vline(const tIndex i) const { return _vln[i]; }

  int resX() const { return _resX; }
  int resY() const { return _resY; }


private:

void buildNeighbor() {
    for (auto &cell : _pidxInGrid) {   // Clear existing data in the grid
        cell.clear();
    }
 // register each particle to a cell
    for (tIndex p = 0; p < _pos.size(); p++) {
      tIndex idx = idx1d(floor(_pos[p].x), floor(_pos[p].y));

      // the particle p will be associated to cell idx:
      _pidxInGrid[idx].push_back(p);
    }
}

void buildPredictedNeighbor() { // with predictedPos ?
    for (auto &cell : _pidxInGrid) {
        cell.clear();
    }
    for (tIndex p = 0; p < _predictedPos.size(); p++) {
      tIndex idx = idx1d(floor(_predictedPos[p].x), floor(_predictedPos[p].y));

      _pidxInGrid[idx].push_back(p);
    }
}

std::vector<tIndex> getNeighbors(const Vec2f& pos) const {
    int x = floor(pos.x);
    int y = floor(pos.y);
    std::vector<tIndex> neigbors;

    neigbors.push_back(idx1d(x, y));                

    if (x > 0) {
      neigbors.push_back(idx1d(x-1, y));

      if (y > 0)
        neigbors.push_back(idx1d(x-1, y-1));

      if (y < _resY-1)
        neigbors.push_back(idx1d(x-1, y+1));
    }

    if (x< _resX-1) {
      neigbors.push_back(idx1d(x+1, y));

      if (y > 0)
        neigbors.push_back(idx1d(x+1, y-1));

      if (y < (_resY-1))
        neigbors.push_back(idx1d(x+1, y+1));
    }

    if (y >0) {
      neigbors.push_back(idx1d(x, y-1));
    }

    if (y< _resY-1) {
      neigbors.push_back(idx1d(x, y+1));
    }


    return neigbors;
}



  void applyBodyForce() {
    for (tIndex i=0; i<_pos.size(); i++){
      _acc[i]+=_g;
    }
  }



  void applyViscousForce()
  {
    Vec2f num = Vec2f(0,0), den = Vec2f(0,0);                
    Vec2f relPos = Vec2f(0,0);

    for (tIndex i = 0; i < _pos.size(); i++) {
      std::vector<tIndex> neighCells = getNeighbors(_pos[i]);

      for (auto &n : neighCells) {      
        std::vector<tIndex> neighPartCell_n = _pidxInGrid[n];

        for (auto &p : neighPartCell_n) {                  
          if (i != p)
            _acc[i] += (2 * _nu * _m0 / _d[p] * (_vel[p] - _vel[i]) * (_pos[i] - _pos[p]) * _kernel.grad_w(_pos[i] - _pos[p])) / ((_pos[i] - _pos[p]) * (_pos[i] - _pos[p]) + 0.01 * _h * _h);
        } //nul a i=0 car vel=0 donc ok
      }
    }
  }
  

  void initialization(){
    for (tIndex i=0; i<_pos.size(); i++){
      _p[i] = 0;
      // and Fp init = 0 so nothing to add to _acc
    }
  }


void predictVelocity(){
  for (tIndex k = 0; k < _pos.size(); k++)
    _predictedVel[k] = _vel[k] + _dt*(_acc[k]+_acc_p[k]);
}

void predictPosition(){
  for (tIndex k = 0; k < _pos.size(); k++)
    _predictedPos[k] = _pos[k] + _dt*_predictedVel[k];
}



  void predictDensity(){  //computeDensity but with the predicted data
    for (tIndex i = 0; i < _predictedPos.size(); i++) {
      _predictedDensity[i] = 0.0;                         
      std::vector<tIndex> neighCells = getNeighbors(_predictedPos[i]);   // neighbors cells

      for (auto &n : neighCells) {
        std::vector<tIndex> neighPartCell_n = _pidxInGrid[n];    

        for (auto &p : neighPartCell_n)
          _predictedDensity[i] += _m0 * _kernel.w(_predictedPos[i] - _predictedPos[p]);
      }
    }
  }

  void calculateDensityError(){
    for (tIndex i = 0; i < _predictedPos.size(); i++) {
      _dErr[i]=_predictedDensity[i] - _d0;
    }
  }


  void calculatePressure(){
    for (tIndex i = 0; i < _pos.size(); i++) {
      _p[i] += _dErr[i]*delta; //ok
    }
  }


  void applyPressureForce()
  {
    for (tIndex i = 0; i < _pos.size(); i++) {
      std::vector<tIndex> neighCells = getNeighbors(_predictedPos[i]);

      for (auto &n : neighCells) {
        std::vector<tIndex> neighPartCell_n = _pidxInGrid[n];

        for (auto &p : neighPartCell_n) {
          if (i != p)
            _acc[i] -= _m0 * 2 * (_p[i] /(_d0*_d0)) * _kernel.grad_w(_predictedPos[i] - _predictedPos[p]);
       }
      }
    }
  }


//uptade velocity and position of cells
   void updateVelocity()
  {
    for (tIndex k = 0; k < _pos.size(); k++)
      _vel[k] = _vel[k] + _dt*_acc[k];
  }

void updatePosition() {
    for (size_t i = 0; i < _pos.size(); ++i) {
       
            _pos[i] += _dt * _vel[i];
    }
}



//resolve collision between particles
  void resolveCollision()
  {
    std::vector<tIndex> need_res;
    for(tIndex i=0; i<particleCount(); ++i) {
      if(_pos[i].x<_l || _pos[i].y<_b || _pos[i].x>_r || _pos[i].y>_t)
        need_res.push_back(i);
    }

    for(
      std::vector<tIndex>::const_iterator it=need_res.begin();
      it<need_res.end();
      ++it) {
      const Vec2f p0 = _pos[*it];
      _pos[*it].x = clamp(_pos[*it].x, _l, _r);
      _pos[*it].y = clamp(_pos[*it].y, _b, _t);
      _vel[*it] = (_pos[*it] - p0)/_dt;
    }
  }

  void updateColor()
  {
    for(tIndex i=0; i<particleCount(); ++i) {
      _col[i*4+0] = 0.6;
      _col[i*4+1] = 0.6;
      _col[i*4+2] = _d[i]/_d0;
    }
  }



  void updateVelLine()
  {
    for(tIndex i=0; i<particleCount(); ++i) {
      _vln[i*4+0] = _pos[i].x;
      _vln[i*4+1] = _pos[i].y;
      _vln[i*4+2] = _pos[i].x + _vel[i].x;
      _vln[i*4+3] = _pos[i].y + _vel[i].y;
    }
  }

  Real calculateAverage(const std::vector<Real>& vec) {
    Real sum = 0.0;
    for (const Real& element : vec) {
        sum += element;
    }

    return sum / vec.size();
}

  inline tIndex idx1d(const int i, const int j) const {
    return i + j * resX();
}


  const CubicSpline _kernel;

  // particle data
  std::vector<Vec2f> _pos;      // position
    std::vector<Vec2f> _particulesVirtuelles;
  std::vector<Vec2f> _predictedPos; //predicted positions
  std::vector<Vec2f> _predictedVel; //predicted velocity
  std::vector<Vec2f> _vel;      // velocity
  std::vector<Vec2f> _acc;      // acceleration
  std::vector<Vec2f> _acc_p; 
  std::vector<Real>  _p;        // pressure
  std::vector<Real>  _Pa;        // pressure
  std::vector<Real>  _d;        // density
  std::vector<Real>  _predictedDensity; //predicted Density
  std::vector<Real>  _dErr;     // density - d0

  std::vector< std::vector<tIndex> > _pidxInGrid; // will help you find neighbor particles

  std::vector<float> _col;    // particle color; just for visualization
  std::vector<float> _vln;    // particle velocity lines; just for visualization

  // simulation
  Real _dt;                     // time step
  Real beta;
  int _resX, _resY;             // background grid resolution

  // wall
  Real _l, _r, _b, _t;          // wall (boundary)

  // SPH coefficients
  Real _nu;                     // viscosity coefficient
  Real _d0;                     // rest density
  Real _h;                      // particle spacing (i.e., diameter)
  Vec2f _g;                     // gravity

  Real _m0;                     // rest mass
  Real _k;                      // EOS coefficient

  Real _eta;
  Real _c;                      // speed of sound
  Real _gamma;                  // EOS power factor
};

SphSolver gSolver(0.08, 0.5, 1e3, Vec2f(0, -9.8), 0.01, 7.0);

void printHelp()
{
  std::cout <<
    "> Help:" << std::endl <<
    "    Keyboard commands:" << std::endl <<
    "    * H: print this help" << std::endl <<
    "    * P: toggle simulation" << std::endl <<
    "    * G: toggle grid rendering" << std::endl <<
    "    * V: toggle velocity rendering" << std::endl <<
    "    * S: save current frame into a file" << std::endl <<
    "    * Q: quit the program" << std::endl;
}

// Executed each time the window is resized. Adjust the aspect ratio and the rendering viewport to the current window.
void windowSizeCallback(GLFWwindow *window, int width, int height)
{
  gWindowWidth = width;
  gWindowHeight = height;
  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

// Executed each time a key is entered.
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if(action == GLFW_PRESS && key == GLFW_KEY_H) {
    printHelp();
  } else if(action == GLFW_PRESS && key == GLFW_KEY_S) {
    gSaveFile = !gSaveFile;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_G) {
    gShowGrid = !gShowGrid;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_V) {
    gShowVel = !gShowVel;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_P) {
    gAppTimerStoppedP = !gAppTimerStoppedP;
    if(!gAppTimerStoppedP)
      gAppTimerLastClockTime = static_cast<float>(glfwGetTime());
  } else if(action == GLFW_PRESS && key == GLFW_KEY_Q) {
    glfwSetWindowShouldClose(window, true);
  }
}

void initGLFW()
{
  // Initialize GLFW, the library responsible for window management
  if(!glfwInit()) {
    std::cerr << "ERROR: Failed to init GLFW" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Before creating the window, set some option flags
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // only if requesting 3.0 or above
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE); // for OpenGL below 3.2
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

  // Create the window
  gWindowWidth = gSolver.resX()*kViewScale;
  gWindowHeight = gSolver.resY()*kViewScale;
  gWindow = glfwCreateWindow(
    gSolver.resX()*kViewScale, gSolver.resY()*kViewScale,
    "Basic SPH Simulator", nullptr, nullptr);
  if(!gWindow) {
    std::cerr << "ERROR: Failed to open window" << std::endl;
    glfwTerminate();
    std::exit(EXIT_FAILURE);
  }

  // Load the OpenGL context in the GLFW window
  glfwMakeContextCurrent(gWindow);

  // not mandatory for all, but MacOS X
  glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);

  // Connect the callbacks for interactive control
  glfwSetWindowSizeCallback(gWindow, windowSizeCallback);
  glfwSetKeyCallback(gWindow, keyCallback);

  std::cout << "Window created: " <<
    gWindowWidth << ", " << gWindowHeight << std::endl;
}

void clear();

void exitOnCriticalError(const std::string &message)
{
  std::cerr << "> [Critical error]" << message << std::endl;
  std::cerr << "> [Clearing resources]" << std::endl;
  clear();
  std::cerr << "> [Exit]" << std::endl;
  std::exit(EXIT_FAILURE);
}

void initOpenGL()
{
  // Load extensions for modern OpenGL
  if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    exitOnCriticalError("[Failed to initialize OpenGL context]");

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void init()
{
  gSolver.initScene(48, 32, 16, 16);
  delta = gSolver.preComputeDelta(); //delta ok
          std::cout << delta << std::flush;
        std::cout << "\n" << std::flush;

  initGLFW();                   // Windowing system
  initOpenGL();
}

void clear()
{
  glfwDestroyWindow(gWindow);
  glfwTerminate();
}

// The main rendering call
void render()
{
  glClearColor(.4f, .4f, .4f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // grid guides
  if(gShowGrid) {
    glBegin(GL_LINES);
    for(int i=1; i<gSolver.resX(); ++i) {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), 0.0);
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()));
    }
    for(int j=1; j<gSolver.resY(); ++j) {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(0.0, static_cast<Real>(j));
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j));
    }
    glEnd();
  }

  // render particles
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glPointSize(0.25f*kViewScale);

  glColorPointer(4, GL_FLOAT, 0, &gSolver.color(0));
  glVertexPointer(2, GL_FLOAT, 0, &gSolver.position(0));
  glDrawArrays(GL_POINTS, 0, gSolver.particleCount());

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // velocity
  if(gShowVel) {
    glColor4f(0.0f, 0.0f, 0.5f, 0.2f);

    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer(2, GL_FLOAT, 0, &gSolver.vline(0));
    glDrawArrays(GL_LINES, 0, gSolver.particleCount()*2);

    glDisableClientState(GL_VERTEX_ARRAY);
  }

  if(gSaveFile) {
    std::stringstream fpath;
    fpath << "s" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";

    std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
    const short int w = gWindowWidth;
    const short int h = gWindowHeight;
    std::vector<int> buf(w*h*3, 0);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, &(buf[0]));

    FILE *out = fopen(fpath.str().c_str(), "wb");
    short TGAhead[] = {0, 2, 0, 0, 0, 0, w, h, 24};
    fwrite(&TGAhead, sizeof(TGAhead), 1, out);
    fwrite(&(buf[0]), 3*w*h, 1, out);
    fclose(out);
    gSaveFile = false;

    std::cout << "Done" << std::endl;
  }
}

// Update any accessible variable based on the current time
void update(const float currentTime)
{
  if(!gAppTimerStoppedP) {
    // NOTE: When you want to use application's dt ...
    // const float dt = currentTime - gAppTimerLastClockTime;
    // gAppTimerLastClockTime = currentTime;
    // gAppTimer += dt;

    // solve 10 steps
    for(int i=0; i<1; ++i) gSolver.update();
  }
}

int main(int argc, char **argv)
{
  init();
  while(!glfwWindowShouldClose(gWindow)) {
    update(static_cast<float>(glfwGetTime()));
    render();
    glfwSwapBuffers(gWindow);
    glfwPollEvents();
  }
  clear();
  std::cout << " > Quit" << std::endl;
  return EXIT_SUCCESS;
}
