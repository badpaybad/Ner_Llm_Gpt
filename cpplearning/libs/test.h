#include <iostream>

class TestClass{
    private:
        void _funcPrivate();

    public:
        TestClass(std::string name){            
            Name = name;
        };
        std::string Name;

        void ToString(){
            std::cout << Name <<"\n";
        };

        void funcPublic();
};