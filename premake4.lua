--OPENCL="/opt/AMDAPP/"
OPENCL="/usr/local/cuda"
OZLIB="OZlib"
-- Creates a couple of folders if they don't exist already
os.mkdir('images/Results')
os.mkdir('images/temp_results')

solution "SignedDistFunc"
 --configurations { "Debug", "Release" }
 configurations { "Release" }
   --location("build")

   -- A project defines one build target
   project "SignedDistFunc"
   targetdir "dist"

        --Shared are .o static are .a
      kind "ConsoleApp"

      includedirs{--My libraries
                  OZLIB,
                  -- cl.h
                 OPENCL.."/include",
                 "src", "src/headers"
             }

      -- os.copyfile("src/resources/SDF.cl","dist")
      -- os.copyfile("src/resources/SDFVoroBuf.cl","dist")

      libdirs{OZLIB}

      location "."
      language "C++"

      -- Current project files
      files {"src/**.h", "src/**.cpp" }     

      links({"OpenCL","GL","GLU","glut","GLEW","X11","m","FileManager",
          "GLManager","CLManager","ImageManager","GordonTimers","freeimage"})
 
      configuration "Debug"
         defines { "DEBUG" , "PRINT" }
         defines { "DEBUG" }
         flags { "Symbols" }


      configuration "Release"
         defines { "NDEBUG" }
         flags { "Optimize" }
