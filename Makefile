CXX := icpx
OPT_FLAGS := -O3
SYCL_FLAGS := -fsycl
INCLUDE_FLAGS := $$(python3 -m pybind11 --includes)
LIB_NAME := onemath_v2c$$(python3-config --extension-suffix)
OUTPUT_FLAGS := -shared -fPIC -o $(LIB_NAME)
LINK_FLAGS := -lonemath
SOURCES := onemath_v2c.cpp

onemath_v2c:
	$(CXX) $(OPT_FLAGS) $(SYCL_FLAGS) $(INCLUDE_FLAGS) $(OUTPUT_FLAGS) $(LINK_FLAGS) $(SOURCES)

clean:
	rm -f $(LIB_NAME)
