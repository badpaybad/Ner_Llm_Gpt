#include <iostream>
#include "test.h"

void TestClass::funcPublic(){
    std::cout << "TestClass::funcPublic() with name: "<< this->Name <<"\n";
}
void TestClass::_funcPrivate(){
    std::cout << "TestClass::_funcPrivate()\n" << Name <<"\n";
}