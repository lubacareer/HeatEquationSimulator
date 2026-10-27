// Frozen-field, paired-primary Geant4 transport. No dose surrogate is used.
// All JSON/binary interchange uses metres, tesla, MeV/nucleon, kg and Gy.
#include <G4RunManager.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4UserEventAction.hh>
#include <G4UserSteppingAction.hh>
#include <G4ParticleGun.hh>
#include <G4IonTable.hh>
#include <G4ParticleTable.hh>
#include <G4Proton.hh>
#include <G4NistManager.hh>
#include <G4Material.hh>
#include <G4Box.hh>
#include <G4Orb.hh>
#include <G4Sphere.hh>
#include <G4Torus.hh>
#include <G4Tubs.hh>
#include <G4IntersectionSolid.hh>
#include <G4LogicalVolume.hh>
#include <G4PVPlacement.hh>
#include <G4PVParameterised.hh>
#include <G4PhantomParameterisation.hh>
#include <G4MagneticField.hh>
#include <G4FieldManager.hh>
#include <G4TransportationManager.hh>
#include <G4PropagatorInField.hh>
#include <G4Exception.hh>
#include <G4VModularPhysicsList.hh>
#include <G4PhysListFactory.hh>
#include <G4EmStandardPhysics_option4.hh>
#include <G4Step.hh>
#include <G4Event.hh>
#include <G4Track.hh>
#include <G4SystemOfUnits.hh>
#include <G4PhysicalConstants.hh>
#include <G4Version.hh>
#include <Randomize.hh>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;
using Vec = std::array<double, 3>;
namespace fs = std::filesystem;
Vec sub(Vec a, Vec b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
Vec cross(Vec a, Vec b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
double dot(Vec a,Vec b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
G4ThreeVector gvec(Vec v) { return {v[0]*m,v[1]*m,v[2]*m}; }

struct Tet { std::array<int64_t,4> ids; Vec lo, hi; };
struct Node { Vec lo,hi; int left=-1,right=-1,begin=0,end=0; };
struct FieldLimits {double support_radius=0,world_radius=0,clearance=0,maximum_step=0;};
struct Primary {std::array<double,9> values;};
class FrozenMagneticField: public G4MagneticField {
  std::vector<Vec> points,values;
  std::vector<Tet> tetra;
  std::vector<int> order;
  std::vector<Node> nodes;
  bool cell_constant=false;
  bool exactly_zero=false;
  FieldLimits limits;
  mutable double maximum_query_radius_squared=0.;
  mutable uint64_t field_queries=0;
  int build(int begin,int end) {
    Node node;
    node.begin=begin; node.end=end;
    node.lo={1e100,1e100,1e100};node.hi={-1e100,-1e100,-1e100};
    for(int k=begin;k<end;k++)for(int j=0;j<3;j++) {
      node.lo[j]=std::min(node.lo[j],tetra[order[k]].lo[j]);
      node.hi[j]=std::max(node.hi[j],tetra[order[k]].hi[j]);
    }
    int id=nodes.size();nodes.push_back(node);
    if(end-begin>12) {
      int axis=0;for(int j=1;j<3;j++)if(node.hi[j]-node.lo[j]>node.hi[axis]-node.lo[axis])axis=j;
      int mid=(begin+end)/2;
      std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,[&](int a,int b){
        return tetra[a].lo[axis]+tetra[a].hi[axis]<tetra[b].lo[axis]+tetra[b].hi[axis];});
      nodes[id].left=build(begin,mid);nodes[id].right=build(mid,end);
    }
    return id;
  }
  bool find(int n,Vec p,Vec& b) const {
    const auto& node=nodes[n];
    for(int j=0;j<3;j++)if(p[j]<node.lo[j]-1e-9||p[j]>node.hi[j]+1e-9)return false;
    if(node.left>=0)return find(node.left,p,b)||find(node.right,p,b);
    for(int k=node.begin;k<node.end;k++) {
      const auto& t=tetra[order[k]];
      bool outside=false;for(int j=0;j<3;j++)if(p[j]<t.lo[j]-1e-9||p[j]>t.hi[j]+1e-9)outside=true;
      if(outside)continue;
      Vec a=sub(points[t.ids[1]],points[t.ids[0]]), c=sub(points[t.ids[2]],points[t.ids[0]]);
      Vec d=sub(points[t.ids[3]],points[t.ids[0]]), q=sub(p,points[t.ids[0]]);
      double determinant=dot(a,cross(c,d));
      if(std::abs(determinant)<1e-30)continue;
      std::array<double,4> w;
      w[1]=dot(q,cross(c,d))/determinant;w[2]=dot(a,cross(q,d))/determinant;
      w[3]=dot(a,cross(c,q))/determinant;w[0]=1-w[1]-w[2]-w[3];
      if(*std::min_element(w.begin(),w.end()) < -1e-8)continue;
      if(cell_constant){b=values[order[k]];return true;}
      b={0,0,0};for(int i=0;i<4;i++)for(int j=0;j<3;j++)b[j]+=w[i]*values[t.ids[i]][j];
      return true;
    }
    return false;
  }
public:
  bool enabled=true;
  explicit FrozenMagneticField(const std::string& filename) {
    std::ifstream in(filename,std::ios::binary);int64_t count[3];
    in.read(reinterpret_cast<char*>(count),sizeof(count));
    if(!in||count[0]<4||count[1]<1)throw std::runtime_error("Invalid frozen exterior field file");
    if(count[2]!=0&&count[2]!=1)throw std::runtime_error("Unsupported frozen magnetic field storage mode");
    cell_constant=count[2]==1;
    points.resize(count[0]);values.resize(cell_constant?count[1]:count[0]);tetra.resize(count[1]);
    in.read(reinterpret_cast<char*>(points.data()),points.size()*sizeof(Vec));
    for(auto& t:tetra)in.read(reinterpret_cast<char*>(t.ids.data()),sizeof(t.ids));
    in.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(Vec));
    if(!in)throw std::runtime_error("Truncated frozen exterior field file");
    for(const auto& point:points)for(double value:point)if(!std::isfinite(value))
      throw std::runtime_error("Frozen exterior field coordinates must be finite");
    exactly_zero=true;
    for(const auto& value:values)for(double component:value){
      if(!std::isfinite(component))throw std::runtime_error("Frozen magnetic field values must be finite");
      exactly_zero=exactly_zero&&component==0.;
    }
    for(auto& t:tetra)for(int j=0;j<3;j++){
      t.lo[j]=1e100;t.hi[j]=-1e100;
      for(auto i:t.ids){if(i<0||i>=count[0])throw std::runtime_error("Invalid tetrahedron vertex");
        t.lo[j]=std::min(t.lo[j],points[i][j]);t.hi[j]=std::max(t.hi[j],points[i][j]);}
    }
    order.resize(tetra.size());std::iota(order.begin(),order.end(),0);build(0,order.size());
  }
  bool is_exactly_zero()const{return exactly_zero;}
  double maximum_query_radius()const{return std::sqrt(maximum_query_radius_squared);}
  uint64_t query_count()const{return field_queries;}
  const FieldLimits& field_limits()const{return limits;}
  const FieldLimits& validate_world(double world_radius,double requested_step=0.){
    if(!std::isfinite(world_radius)||world_radius<=0.)throw std::runtime_error("Transport world radius must be finite and positive");
    struct FaceInfo {int count=0;double first_side=0.;};
    std::map<std::array<int64_t,3>,FaceInfo> faces;
    for(const auto& t:tetra){
      double determinant=dot(sub(points[t.ids[1]],points[t.ids[0]]),
          cross(sub(points[t.ids[2]],points[t.ids[0]]),sub(points[t.ids[3]],points[t.ids[0]])));
      if(!std::isfinite(determinant)||std::abs(determinant)<1e-30)
        throw std::runtime_error("Frozen exterior map contains a degenerate tetrahedron");
      for(int missing=0;missing<4;missing++){
        std::array<int64_t,3> face;int next=0;
        for(int vertex=0;vertex<4;vertex++)if(vertex!=missing)face[next++]=t.ids[vertex];
        std::sort(face.begin(),face.end());
        auto& info=faces[face];
        if(++info.count>2)throw std::runtime_error("Frozen exterior map has a nonmanifold tetrahedral face");
        Vec normal=cross(sub(points[face[1]],points[face[0]]),sub(points[face[2]],points[face[0]]));
        double side=dot(normal,sub(points[t.ids[missing]],points[face[0]]));
        if(!std::isfinite(side)||side==0.)throw std::runtime_error("Frozen exterior map has an invalid tetrahedral face");
        if(info.count==1)info.first_side=side;
        else if((info.first_side>0.)==(side>0.))
          throw std::runtime_error("Frozen exterior map has incident tetrahedra on the same side of a shared face");
      }
    }
    Vec origin_value;
    if(!find(0,{0.,0.,0.},origin_value))throw std::runtime_error("The accepted exterior map does not contain the world origin");
    double support=std::numeric_limits<double>::infinity();
    for(const auto& [face,info]:faces)if(info.count==1){
      Vec normal=cross(sub(points[face[1]],points[face[0]]),sub(points[face[2]],points[face[0]]));
      support=std::min(support,std::abs(dot(normal,points[face[0]]))/std::sqrt(dot(normal,normal)));
    }
    double clearance=support-world_radius;
    if(!std::isfinite(support)||clearance<=std::max(1e-9,support*1e-10))
      throw std::runtime_error("The transport world is not enclosed with positive clearance by the accepted FE tetrahedral map; enlarge the exterior domain and rerun");
    double maximum_step=std::min(.1,clearance/4);
    if(requested_step!=0.){
      if(!std::isfinite(requested_step)||requested_step<=0.||requested_step>maximum_step)
        throw std::runtime_error("Requested magnetic propagation step must be positive and no greater than the accepted-map clearance limit");
      maximum_step=requested_step;
    }
    limits={support,world_radius,clearance,maximum_step};return limits;
  }
  void GetFieldValue(const G4double p[4],G4double* b) const override {
    if(!enabled){b[0]=b[1]=b[2]=0;return;}
    Vec location={p[0]/m,p[1]/m,p[2]/m};
    bool finite=std::isfinite(location[0])&&std::isfinite(location[1])&&std::isfinite(location[2]);
    ++field_queries;
    if(finite)maximum_query_radius_squared=std::max(maximum_query_radius_squared,dot(location,location));
    Vec out;
    if(!finite||!find(0,location,out)){
      G4ExceptionDescription message;
      message<<std::setprecision(17)<<"Transport requested a field outside the accepted FE exterior mesh. "
        <<"Query [m] = ("<<p[0]/m<<", "<<p[1]/m<<", "<<p[2]/m<<"); world radius [m] = "<<limits.world_radius
        <<"; certified support radius [m] = "<<limits.support_radius<<"; maximum propagation step [m] = "<<limits.maximum_step
        <<". Enlarge the exterior domain and rerun.";
      G4Exception("FrozenMagneticField","OutsideAcceptedMap",FatalException,message);
    }
    for(int j=0;j<3;j++)b[j]=out[j]*tesla;
  }
};

std::vector<Primary> load_primaries(const json& request,double world_radius){
  int histories=request.at("config").at("radiation_histories");
  if(histories<2)throw std::runtime_error("At least two primary histories are needed for dose uncertainty");
  std::ifstream input(std::string(request.at("primary_file")),std::ios::binary);
  std::vector<Primary> primaries;Primary primary;
  while(input.read(reinterpret_cast<char*>(primary.values.data()),sizeof(primary.values))){
    const auto& v=primary.values;
    for(double value:v)if(!std::isfinite(value))throw std::runtime_error("Incident primary samples must be finite");
    if(v[0]<1||v[0]>28||v[1]<v[0]||std::floor(v[0])!=v[0]||std::floor(v[1])!=v[1]||v[2]<=0.)
      throw std::runtime_error("Incident primary isotope and energy are invalid");
    if(v[3]*v[3]+v[4]*v[4]+v[5]*v[5]>=world_radius*world_radius)
      throw std::runtime_error("Incident primary starts outside the validated transport world");
    if(std::abs(v[6]*v[6]+v[7]*v[7]+v[8]*v[8]-1.)>1e-6)
      throw std::runtime_error("Incident primary direction must be a unit vector");
    primaries.push_back(primary);
  }
  if(input.bad()||input.gcount()!=0||primaries.size()!=size_t(histories))
    throw std::runtime_error("Paired primary file has wrong event count or a truncated record");
  return primaries;
}

json field_limit_metadata(const FrozenMagneticField& field){
  const auto& limits=field.field_limits();
  return {{"field_map_support_radius_m",limits.support_radius},{"world_radius_m",limits.world_radius},
    {"field_map_clearance_m",limits.clearance},{"maximum_field_step_m",limits.maximum_step},
    {"maximum_field_query_radius_m",field.maximum_query_radius()},{"magnetic_field_queries",field.query_count()},
    {"exact_zero_field_fast_path",field.is_exactly_zero()},
    {"field_transport_mode",field.is_exactly_zero()?"straight-exact-zero":"bounded-frozen-map"}};
}

struct Target {
  std::array<int,3> shape{16,16,16};
  Vec spacing{};
  std::vector<size_t> material_indices;
  std::vector<int> organ_ids;
  std::vector<G4Material*> materials;
  std::map<int,std::string> organ_names;
  std::map<int,double> organ_mass;
  std::vector<double> voxel_mass;
  G4LogicalVolume* voxel_logical=nullptr;
  size_t size()const{return size_t(shape[0])*shape[1]*shape[2];}
  Vec half_size()const{return {shape[0]*spacing[0]/2,shape[1]*spacing[1]/2,shape[2]*spacing[2]/2};}
  Vec center(size_t index)const{
    auto half=half_size();
    return {(int(index%shape[0])+.5)*spacing[0]-half[0],
            (int(index/shape[0]%shape[1])+.5)*spacing[1]-half[1],
            (int(index/(shape[0]*shape[1]))+.5)*spacing[2]-half[2]};
  }
};

Target load_target(const json& request) {
  Target t;auto config=request.at("config");std::string target=config.at("target");
  auto* nist=G4NistManager::Instance();
  if(target=="water") {
    double extent=std::min(.20,double(config.at("habitat_radius"))*.8/std::sqrt(3.));
    t.spacing={2*extent/16,2*extent/16,2*extent/16};
    t.materials.push_back(nist->FindOrBuildMaterial("G4_WATER"));
    t.material_indices.assign(t.size(),0);t.organ_ids.assign(t.size(),1);t.organ_names[1]="Water target";
  } else {
    bool male=target=="male";std::string prefix=male?"AM":"AF";
    fs::path root=request.at("phantom_path");
    fs::path raw=root/"ICRP110_g4dat"/"P110_data_V1.2"/prefix;
    if(!fs::exists(raw/(prefix+"_organs.dat")))throw std::runtime_error(
      "ICRP110 data missing. Run setup_geant4.sh --phantom-data or set phantom_path to the official ICRPdata directory.");
    t.shape=male?std::array<int,3>{254,127,222}:std::array<int,3>{299,137,348};
    t.spacing=male?Vec{.002137,.002137,.008}:Vec{.001775,.001775,.00484};
    const std::array<int,13> atomic{1,6,7,8,11,12,15,16,17,19,20,26,53};
    std::map<int,std::array<double,13>> media;
    std::ifstream media_file(raw/(prefix+"_media.dat"));std::string line;
    std::regex medium_re(R"(^\s*(\d+)\s+(.+?)\s+((?:[0-9.]+\s+){12}[0-9.]+)\s*$)");
    while(std::getline(media_file,line)){
      std::smatch match;if(!std::regex_match(line,match,medium_re))continue;
      std::istringstream numbers(match[3]);std::array<double,13> fractions{};
      for(auto& f:fractions)numbers>>f;media[std::stoi(match[1])]=fractions;
    }
    std::ifstream organs(raw/(prefix+"_organs.dat"));
    std::regex organ_re(R"(^\s*(\d+)\s+(.+?)\s+(\d+)\s+([0-9.]+)\s*$)");
    std::map<int,size_t> organ_material;
    t.materials.push_back(nist->FindOrBuildMaterial("G4_AIR"));organ_material[0]=0;
    while(std::getline(organs,line)){
      std::smatch match;if(!std::regex_match(line,match,organ_re))continue;
      int id=std::stoi(match[1]),medium=std::stoi(match[3]);double density=std::stod(match[4]);
      if(!media.count(medium))throw std::runtime_error("ICRP110 organ references an unavailable tissue composition");
      auto fractions=media.at(medium);double total=std::accumulate(fractions.begin(),fractions.end(),0.);
      int elements=std::count_if(fractions.begin(),fractions.end(),[](double f){return f>0;});
      auto* material=new G4Material(prefix+"_organ_"+std::to_string(id),density*g/cm3,elements);
      for(int j=0;j<13;j++)if(fractions[j]>0)material->AddElement(nist->FindOrBuildElement(atomic[j]),fractions[j]/total);
      organ_material[id]=t.materials.size();t.materials.push_back(material);t.organ_names[id]=match[2];
    }
    if(media.size()!=53||t.organ_names.size()<140)throw std::runtime_error("Incomplete ICRP110 material or organ table");
    // Official Geant4 ICRP110 slices add label 141 at the two end caps and
    // explicitly map it to skin medium 27 (same composition/density as 122).
    // https://github.com/Geant4/geant4/tree/master/examples/advanced/ICRP110_HumanPhantoms
    organ_material[141]=organ_material.at(122);
    t.organ_names[141]="Skin end caps (ICRP110 voxel boundary)";
    t.organ_ids.resize(t.size());t.material_indices.resize(t.size());
    size_t offset=0;
    for(int slice=1;slice<=t.shape[2];slice++){
      std::ifstream input(root/"ICRP110_g4dat"/prefix/(prefix+"_slice"+std::to_string(slice)+".g4dat"));
      int nx,ny,nz;input>>nx>>ny>>nz;
      if(!input||nx!=t.shape[0]||ny!=t.shape[1]||nz!=t.shape[2])throw std::runtime_error("ICRP110 slice header is missing or inconsistent");
      for(int k=0;k<nx*ny;k++,offset++){
        int organ;input>>organ;if(!input||!organ_material.count(organ))throw std::runtime_error("ICRP110 slice has an unknown or missing organ ID");
        t.organ_ids[offset]=organ;t.material_indices[offset]=organ_material.at(organ);
      }
    }
  }
  auto h=t.half_size();double habitat=config.at("habitat_radius");
  if(std::sqrt(dot(h,h))>=habitat)throw std::runtime_error(
    "The selected anatomical phantom does not fit inside the spherical habitat. Increase habitat_radius and torus major_radius and rerun.");
  t.voxel_mass.resize(t.size());double volume=t.spacing[0]*t.spacing[1]*t.spacing[2]*m3;
  for(size_t i=0;i<t.size();i++){
    t.voxel_mass[i]=t.materials[t.material_indices[i]]->GetDensity()*volume/kg;
    if(t.organ_ids[i]>0&&t.organ_ids[i]!=140)t.organ_mass[t.organ_ids[i]]+=t.voxel_mass[i];
  }
  return t;
}

class Detector: public G4VUserDetectorConstruction {
  json request;
public:
  Target target;
  FrozenMagneticField* field;
  Detector(json r,FrozenMagneticField* f):request(std::move(r)),field(f){}
  G4VPhysicalVolume* Construct()override{
    auto config=request.at("config");auto* nist=G4NistManager::Instance();
    auto* vacuum=nist->FindOrBuildMaterial("G4_Galactic");
    auto* world_solid=new G4Orb("world",double(request.at("world_radius_m"))*m);
    auto* world=new G4LogicalVolume(world_solid,vacuum,"world");
    auto* physical=new G4PVPlacement(nullptr,{},world,"world",nullptr,false,0,true);
    double R=config.at("major_radius"),r=config.at("minor_radius"),wall=config.at("vessel_thickness");
    auto composition=config.at("composition");
    auto* reservoir_material=new G4Material("reservoir",double(config.at("density"))*kg/m3,composition.size());
    for(auto item:composition)reservoir_material->AddElement(nist->FindOrBuildElement(int(item[0])),double(item[1]));
    auto* reservoir=new G4LogicalVolume(new G4Torus("reservoir",0,r*m,R*m,0,twopi),reservoir_material,"reservoir");
    new G4PVPlacement(nullptr,{},reservoir,"reservoir",world,false,0,true);
    if(wall>0){
      auto* vessel=new G4LogicalVolume(new G4Torus("vessel",r*m,(r+wall)*m,R*m,0,twopi),nist->FindOrBuildMaterial("G4_STAINLESS-STEEL"),"vessel");
      new G4PVPlacement(nullptr,{},vessel,"vessel",world,false,0,true);
    }
    // Explicit illustrative circular copper winding, geometrically outside vessel.
    // Coil-current multiplicity is encoded by the magnetic solver independently.
    double wire=request.at("imposed_coil").at("wire_radius_m");
    double coilR=request.at("imposed_coil").at("radius_m");
    auto* coil=new G4LogicalVolume(new G4Torus("coil",0,wire*m,coilR*m,0,twopi),nist->FindOrBuildMaterial("G4_Cu"),"coil");
    new G4PVPlacement(nullptr,G4ThreeVector(0,0,double(request.at("imposed_coil").at("z_m"))*m),coil,"coil",world,false,0,true);
    double habitat=config.at("habitat_radius"),hw=config.at("habitat_thickness");
    auto* habitat_air=new G4LogicalVolume(new G4Orb("habitat_air",habitat*m),nist->FindOrBuildMaterial("G4_AIR"),"habitat_air");
    new G4PVPlacement(nullptr,{},habitat_air,"habitat_air",world,false,0,true);
    if(hw>0){
      auto* shell=new G4Sphere("habitat_wall",habitat*m,(habitat+hw)*m,0,twopi,0,pi);
      auto* logical=new G4LogicalVolume(shell,nist->FindOrBuildMaterial("G4_Al"),"habitat_wall");
      new G4PVPlacement(nullptr,{},logical,"habitat_wall",world,false,0,true);
    }
    if(bool(config.at("ground_enabled"))){
      // A tangent half-space clipped to the spherical world, with an explicit
      // silica composition (illustrative regolith, documented in metadata).
      double wr=request.at("world_radius_m");double top=-std::max(r+wall,habitat+hw)-.01;
      auto* box=new G4Box("ground_halfspace",wr*m,wr*m,wr*m);
      auto* ground=new G4IntersectionSolid("ground",world_solid,box,nullptr,G4ThreeVector(0,0,(top-wr)*m));
      new G4PVPlacement(nullptr,{},new G4LogicalVolume(ground,nist->FindOrBuildMaterial("G4_SILICON_DIOXIDE"),"ground"),"ground",world,false,0,true);
    }
    target=load_target(request);
    auto half=target.half_size();
    auto* envelope_solid=new G4Box("target_container",half[0]*m,half[1]*m,half[2]*m);
    auto* envelope=new G4LogicalVolume(envelope_solid,nist->FindOrBuildMaterial("G4_AIR"),"target_container");
    new G4PVPlacement(nullptr,{},envelope,"target_container",habitat_air,false,0,true);
    auto s=target.spacing;
    auto* voxel=new G4Box("target_voxel",s[0]*m/2,s[1]*m/2,s[2]*m/2);
    target.voxel_logical=new G4LogicalVolume(voxel,target.materials[0],"target_voxel");
    auto* parameter=new G4PhantomParameterisation;
    parameter->SetVoxelDimensions(s[0]*m/2,s[1]*m/2,s[2]*m/2);
    parameter->SetNoVoxels(target.shape[0],target.shape[1],target.shape[2]);
    parameter->SetMaterials(target.materials);parameter->SetMaterialIndices(target.material_indices.data());
    parameter->BuildContainerSolid(envelope_solid);parameter->SetSkipEqualMaterials(false);
    auto* placed=new G4PVParameterised("target_voxel",target.voxel_logical,envelope,kUndefined,target.size(),parameter);
    placed->SetRegularStructureId(1);
    return physical;
  }
  void ConstructSDandField()override{
    auto* manager=G4TransportationManager::GetTransportationManager()->GetFieldManager();
    if(field->is_exactly_zero()){manager->SetDetectorField(nullptr);return;}
    manager->SetDetectorField(field);manager->CreateChordFinder(field);
    manager->SetDeltaOneStep(.01*mm);manager->SetDeltaIntersection(.001*mm);
    G4TransportationManager::GetTransportationManager()->GetPropagatorInField()->SetLargestAcceptableStep(field->field_limits().maximum_step*m);
  }
};

struct Accumulator {
  std::map<int,double> sum,squares,secondary,secondary_squares;
  std::unordered_map<size_t,double> voxel_sum,voxel_squares;
  std::map<int,double> event,event_secondary;
  std::unordered_map<size_t,double> event_voxel;
  int habitat_hits=0,hardware_hits=0,primary_habitat_hits=0,primary_hardware_hits=0;
  bool event_hit=false,event_hardware=false,event_primary_hit=false,event_primary_hardware=false;
  void finish(){
    for(auto [id,value]:event){sum[id]+=value;squares[id]+=value*value;}
    for(auto [id,value]:event_secondary){secondary[id]+=value;secondary_squares[id]+=value*value;}
    for(auto [id,value]:event_voxel){voxel_sum[id]+=value;voxel_squares[id]+=value*value;}
    habitat_hits+=event_hit;hardware_hits+=event_hardware;
    primary_habitat_hits+=event_primary_hit;primary_hardware_hits+=event_primary_hardware;
    event.clear();event_secondary.clear();event_voxel.clear();event_hit=event_hardware=false;
    event_primary_hit=event_primary_hardware=false;
  }
};
struct State {int mode=0;int histories=0;Accumulator score[2];json request;fs::path progress_file;};
class Generator:public G4VUserPrimaryGeneratorAction{
  G4ParticleGun gun{1};std::vector<Primary> primaries;long seed;
public:
  Generator(const json& request,std::vector<Primary> samples):primaries(std::move(samples)),seed(long(request.at("config").at("seed"))){}
  void GeneratePrimaries(G4Event* event)override{
    const auto& v=primaries.at(event->GetEventID()).values;
    // Reset each primary stream so matched field-on/off histories start from
    // the same random stream, regardless of divergent previous secondaries.
    G4Random::setTheSeed((seed+104729L*event->GetEventID())%2147483646L+1);
    G4ParticleDefinition* particle=(int(v[0])==1&&int(v[1])==1)?G4Proton::Definition():G4IonTable::GetIonTable()->GetIon(int(v[0]),int(v[1]),0.);
    if(!particle)throw std::runtime_error("Geant4 does not define requested primary isotope");
    gun.SetParticleDefinition(particle);gun.SetParticleCharge(v[0]*eplus);
    gun.SetParticleEnergy(v[2]*v[1]*MeV);gun.SetParticlePosition({v[3]*m,v[4]*m,v[5]*m});
    gun.SetParticleMomentumDirection({v[6],v[7],v[8]});gun.GeneratePrimaryVertex(event);
  }
};
class Stepper:public G4UserSteppingAction{
  Detector* detector;State* state;
public:
  Stepper(Detector* d,State* s):detector(d),state(s){}
  void UserSteppingAction(const G4Step* step)override{
    if(step->GetTrack()->GetCurrentStepNumber()>1000000)
      throw std::runtime_error("A track exceeded one million steps (possible frozen-field trapping). No dose result was published; cancel or revise this field/source setup.");
    auto* volume=step->GetPreStepPoint()->GetPhysicalVolume();if(!volume)return;
    auto& score=state->score[state->mode];
    auto name=volume->GetName();
    if(name=="reservoir"||name=="vessel"||name=="coil")score.event_hardware=true;
    if(name=="habitat_wall"||name=="habitat_air"||name=="target_container"||name=="target_voxel")score.event_hit=true;
    if(step->GetTrack()->GetParentID()==0){
      if(name=="reservoir"||name=="vessel"||name=="coil")score.event_primary_hardware=true;
      if(name=="habitat_wall"||name=="habitat_air"||name=="target_container"||name=="target_voxel")score.event_primary_hit=true;
    }
    if(volume->GetLogicalVolume()!=detector->target.voxel_logical)return;
    size_t index=step->GetPreStepPoint()->GetTouchableHandle()->GetCopyNumber();
    int organ=detector->target.organ_ids.at(index);if(organ==0||organ==140)return;
    // All tracks score; parentID>0 identifies secondary contribution.
    double energy=step->GetTotalEnergyDeposit()/joule;if(energy<=0)return;
    score.event[organ]+=energy;score.event_voxel[index]+=energy;
    if(step->GetTrack()->GetParentID()>0)score.event_secondary[organ]+=energy;
  }
};
class Events:public G4UserEventAction{
  State* state;
public:
  explicit Events(State* s):state(s){}
  void EndOfEventAction(const G4Event* event)override{
    state->score[state->mode].finish();
    int done=event->GetEventID()+1;
    if(done%std::max(1,state->histories/100)==0){
      json p={{"fraction",(state->mode+double(done)/state->histories)/2},
              {"message",std::string(state->mode?"Field-off":"Field-on")+" transport "+std::to_string(done)+" / "+std::to_string(state->histories)}};
      // The UI may briefly hold this NTFS file open through WSL's Plan9 mount.
      // Replacing an open destination can throw inside Geant4's event callback.
      // Progress is advisory; readers ignore partial JSON and retry. Scientific
      // results use a separate unique path and are still published atomically.
      std::ofstream progress(state->progress_file);progress<<p.dump();
    }
  }
};
double standard_error(double sum,double squares,int n){
  return n>1?std::sqrt(std::max(0.,(squares-sum*sum/n)/(double(n)*(n-1)))):0.;
}
int main(int argc,char** argv){
  G4RunManager* manager=nullptr;
  try{
    bool inspect=argc==3&&std::string(argv[1])=="--inspect-field";
    if(argc!=2&&!inspect)throw std::runtime_error("Usage: torus-dose request.json (or --inspect-field request.json)");
    const char* request_path=argv[inspect?2:1];
    std::ifstream input(request_path);json request;input>>request;
    if(request.at("schema_version")!=1)throw std::runtime_error("Unsupported dose request schema");
    if(!inspect)std::ofstream(fs::path(request_path).parent_path()/"worker.pid")<<getpid();
    FrozenMagneticField field(request.at("field_file"));
    field.validate_world(request.at("world_radius_m"),request.value("maximum_field_step_m",0.));
    auto primaries=load_primaries(request,field.field_limits().world_radius);
    if(inspect){auto report=field_limit_metadata(field);report["primary_count"]=primaries.size();std::cout<<report.dump()<<std::endl;return 0;}
    State state;state.request=request;state.histories=request.at("config").at("radiation_histories");
    if(state.histories<2)throw std::runtime_error("At least two primary histories are needed for dose uncertainty");
    state.progress_file=fs::path(request_path).parent_path()/"progress.json";
    // A failed user callback can leave Geant4 in EventProc; destroying its run
    // manager while unwinding may exit before our useful error is printed.
    // On failure this short-lived worker leaves cleanup to the operating system.
    manager=new G4RunManager();
    auto* detector=new Detector(request,&field);manager->SetUserInitialization(detector);
    G4PhysListFactory factory;std::string list=request.value("physics_list","Shielding");
    if(list!="Shielding"&&list!="QGSP_BIC_HP")throw std::runtime_error("Physics list must be Shielding or QGSP_BIC_HP");
    auto* physics=factory.GetReferencePhysList(list);if(!physics)throw std::runtime_error("Physics list is unavailable");
    physics->ReplacePhysics(new G4EmStandardPhysics_option4);
    physics->SetDefaultCutValue(double(request.at("production_cut_m"))*m);
    manager->SetUserInitialization(physics);manager->SetUserAction(new Generator(request,std::move(primaries)));
    manager->SetUserAction(new Stepper(detector,&state));manager->SetUserAction(new Events(&state));
    manager->Initialize();
    for(int mode=0;mode<2;mode++){state.mode=mode;field.enabled=mode==0;manager->BeamOn(state.histories);}
    double scale=double(request.at("config").at("incident_fluence_m2"))*double(request.at("source_area_m2"));
    json output={{"schema_version",1},{"source_run_id",request.at("source_run_id")},
      {"config_hash",request.at("config_hash")},{"snapshot_index",request.at("snapshot_index")},
      {"histories",state.histories},{"organs",json::array()}};
    auto& target=detector->target;
    for(auto [id,mass]:target.organ_mass){
      json row={{"id",id},{"name",target.organ_names[id]},{"mass_kg",mass}};
      for(int mode=0;mode<2;mode++){
        auto& s=state.score[mode];std::string label=mode?"field_off":"field_on";
        row[label+"_gy"]=s.sum[id]/state.histories/mass*scale;
        row[label+"_standard_error_gy"]=standard_error(s.sum[id],s.squares[id],state.histories)/mass*scale;
      }
      row["secondary_gy"]=state.score[0].secondary[id]/state.histories/mass*scale;
      row["secondary_standard_error_gy"]=standard_error(state.score[0].secondary[id],state.score[0].secondary_squares[id],state.histories)/mass*scale;
      output["organs"].push_back(row);
    }
    // Sparse voxel artifact includes all scored voxels; missing voxels have
    // zero observed deposition, not an inferred physical zero dose.
    fs::path voxel_file=fs::path(request_path).parent_path()/"voxel_dose.bin";
    std::ofstream voxels(voxel_file,std::ios::binary);
    for(auto [index,sum]:state.score[0].voxel_sum){
      auto center=target.center(index);double mass=target.voxel_mass[index];
      std::array<double,7> record{center[0],center[1],center[2],sum/state.histories/mass*scale,
        standard_error(sum,state.score[0].voxel_squares[index],state.histories)/mass*scale,
        double(target.organ_ids[index]),double(index)};
      voxels.write(reinterpret_cast<char*>(record.data()),sizeof(record));
    }
    voxels.close();
    output["metadata"]={{"engine",G4Version},{"physics_list",list+" + EM option 4"},
      {"production_cut_m",request.at("production_cut_m")},{"field_sha256",request.at("field_sha256")},
      {"field_time_s",request.at("field_time_s")},{"source_area_m2",request.at("source_area_m2")},
      {"field_source",request.value("field_source",json("numerical"))},
      {"source_model_id",request.value("source_model_id",json(nullptr))},
      {"source_model_field_sha256",request.value("source_model_field_sha256",json(nullptr))},
      {"source_model_validation_status",request.value("source_model_validation_status",json(nullptr))},
      {"electric_field","zero; frozen magnetic transport"},
      {"field_interpolation","exact element curl of accepted total P1 vector potential"},
      {"incident_fluence_m2",request.at("config").at("incident_fluence_m2")},
      {"source",request.at("source")},
      {"normalization","absorbed Gy for specified planar/isotropic fluence; no biological weighting"},
      {"target",request.at("config").at("target")},{"voxel_binary_file",voxel_file.string()},
      {"voxel_shape",target.shape},{"voxel_spacing_m",target.spacing},
      {"habitat_hits",state.score[0].habitat_hits},{"field_off_habitat_hits",state.score[1].habitat_hits},
      {"hardware_interceptions",state.score[0].hardware_hits},
      {"primary_habitat_hits",state.score[0].primary_habitat_hits},{"primary_hardware_interceptions",state.score[0].primary_hardware_hits},
      {"hit_definition","counts primary histories with any track entering volume; primary-only counts also supplied"},
      {"geometry_assumptions","spherical aluminum habitat with NIST air interior; vacuum exterior; stainless steel vessel; illustrative copper circular winding; silica ground"},
      {"paired_primary_streams",true},{"random_seed",request.at("config").at("seed")},
      {"phantom_provenance","ICRP Publication 110 official Geant4 dataset; cite Menzel et al. 2009 and Large et al. 2020"},
      {"validation_status","research implementation; physics-list, cuts and mesh convergence required"}};
    output["metadata"].update(field_limit_metadata(field));
    fs::path output_path=std::string(request.at("output_file"));
    std::ofstream(output_path.string()+".tmp")<<output.dump(2);
    fs::rename(output_path.string()+".tmp",output_path);
    delete manager;
    return 0;
  }catch(const std::exception& error){std::cerr<<"Dose transport error: "<<error.what()<<std::endl;return 1;}
}
