# basic makefile
.LIBPATTERNS ="'lib%.so lib%.a' "

CXX   = g++
FC		= g77
F90		= f90
RM		= rm -f
CC		= g++

LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared

CXXLIBDIRS	= -L./../../lib/
#CXXLIBS	= -lxx_usb -lm -lusb -Wl,"-rpath=$(here)/../../lib"
CXXLIBS		= -lxx_usb -lrt -lm -lusb -Wl,"-rpath=/home/studenti/Wiener/lib/"
INCLUDEDIRS 	= -I /home/studenti/Wiener/include/

CXXFLAGS      	= -O -Wall -fPIC -g $(INCLUDEDIRS)

CFLAGS		+= $(INCLUDEDIRS) $(SOFLAG)
CFLAGS		+= $(CXXLIBDIRS) $(CXXLIBS)
LDFLAGS		= $(CXXFLAGS)

PET_basic: PET_basic.cpp
	$(CXX) $(CXXFLAGS) PET_basic.cpp \
	$(CXXLIBS) $(CXXLIBDIRS) -o PET_basic 
	@echo "Done"

adc1: adc1.cpp
	$(CXX) $(CXXFLAGS) adc1.cpp \
	$(CXXLIBS) $(CXXLIBDIRS) -o adc1 
	@echo "Done"

tdc-ini: tdc-ini.cpp
	$(CXX) $(CXXFLAGS) tdc-ini.cpp \
	$(CXXLIBS) $(CXXLIBDIRS) -o tdc-ini
	@echo "Done"

tdc1: tdc1.cpp
	$(CXX) $(CXXFLAGS) tdc1.cpp \
	$(CXXLIBS) $(CXXLIBDIRS) -o tdc1
	@echo "Done"

clean:
	$(RM) $(OBJ) 


