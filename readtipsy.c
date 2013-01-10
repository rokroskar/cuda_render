int i,j;
struct dump h;
struct gas_particle gp;
struct dark_particle dp;
struct star_particle sp;
int npart;
XDR xdrs;
FILE *fp;

if(fp = fopen(std::string(filename).c_str(),"r"));
 else {
   fprintf(stderr, "file \'%s\' not found - exiting promptly\n", filename);
   exit(-1);
 }

        
if (bStandard) {
  assert(sizeof(Real)==sizeof(float)); /* Otherwise, this XDR stuff
                                          ain't gonna work */
  xdrstdio_create(&xdrs, fp, XDR_DECODE);
  xdr_header(&xdrs,&h);
 }
 else {
   fread(&h,sizeof(struct dump),1,fp);
 }

if(ptype == GAS) {
  npart = h.nsph;
  
  for(i=0;i<npart;i++) {
    if (bStandard) {
      xdr_gas(&xdrs,&gp);
    }
    else {
      fread(&gp,sizeof(struct gas_particle),1,fp);
    }
    mass[i] = gp.mass;
    x[i] = gp.pos[0];
    y[i] = gp.pos[1];
    z[i] = gp.pos[2]; 
    vx[i] = gp.vel[0];
    vy[i] = gp.vel[1];
    vz[i] = gp.vel[2]; 
    rho[i] = gp.rho;
    temp[i] = gp.temp;
    eps[i] = gp.hsmooth;
    metals[i] = gp.metals;
    phi[i] = gp.phi;      
  }
 }

if(ptype == DARK) {
  npart = h.ndark;
  fprintf(stderr,"here\n");
  for(i=0;i<npart;i++) {
    if (bStandard) {
      xdr_dark(&xdrs,&dp);
    }
    else {
      fread(&dp,sizeof(struct dark_particle),1,fp);
    }
    mass[i] = dp.mass;
    x[i] = dp.pos[0];
    y[i] = dp.pos[1];
    z[i] = dp.pos[2]; 
    vx[i] = dp.vel[0];
    vy[i] = dp.vel[1];
    vz[i] = dp.vel[2]; 
    eps[i] = dp.eps;
    phi[i] = gp.phi;      
  }
 }


fclose(fp);

