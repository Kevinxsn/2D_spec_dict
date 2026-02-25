program assign_peaks

character(len=100) :: peptide_name, PTM
integer :: peptide_len, parent_charge, maxC13
logical :: DoInternalFrag
real*8, dimension(:), allocatable :: mz_arr, PeakDiff
integer, dimension(:,:,:), allocatable :: Frag_INFO
character(len=400), dimension(:), allocatable :: PeakLabel
character(len=400) :: PeakLabelTemp

character(len=1) :: amino, modification
real*8, dimension(:), allocatable :: amino_mass
real*8, dimension(:), allocatable :: peptide_mass_arr
integer, dimension(:), allocatable :: peptide_amino_arr, peptide_PTM_arr
character(len=100), dimension(:), allocatable :: FragLabel, NeuLossLabel
real*8, dimension(:), allocatable :: Fragmz, Neumz
real*8 :: Fragmz_base, PeakLabelThr_Abs, PeakLabelThr_Frac, parent_mass_max !, fragment_max_charge_fact
character(len=100) :: FragLabel_base
character(len=10) :: numC13_char
integer :: NeuLossTypes, NeuLossTypes_extra, min_internal, ind, maxind, charge, frag_start, frag_length, fragment, Total_To_Match
integer :: NeuLossInd, numC13, maxC13_here, label_num, amino_ind, counter, start_counter, All_Amino_Types, All_PTM_Types
integer :: maxcharge_here, maxcharge_here_internal
integer :: MaxLabels
integer, dimension(:,:), allocatable :: Frag_SFC
logical, dimension(:), allocatable :: NeuLossInclude_Combined
logical, dimension(:,:), allocatable :: NeuLossInclude

real*8 ::   neutral_CO = 27.994914619d0
real*8 ::   neutral_NH3 = 17.026549100d0
real*8 ::   neutral_H2O = 18.010564683d0
real*8 ::   neutral_H = 1.0078250319d0
real*8 ::   neutral_HPO3 = 79.967430d0
real*8 ::   hydronium_cation = 19.01784115084d0
real*8 ::   neutral_CONH2 = 18.01056468d0 !18.03436
real*8 ::   proton = 1.007276466812d0

real*8 ::   C13 = 1.003354835d0


NAMELIST /INPUT/ peptide_name, parent_charge, PeakLabelThr_Abs, PeakLabelThr_Frac, DoInternalFrag, maxC13, PTM

!set defaults
PTM='0'                             !post-translation modifications. default value indicates no PTMs; otherwise, must be a string of the same length as peptide_name, '0' indicates no PTM in that position, 
                                    !other PTMs are listed below (most possible PTMs are not currently implemented, but can be easily added)

DoInternalFrag=.true.               !include internal fragments in peak assignment
maxC13=1                            !the maximum number of C13 atoms allowed per fragment (this value is reduced for shorter fragments)
PeakLabelThr_Abs=0.002              !maximum deviation from theory for assigning peak labels (absolute error in Da, used only if this is > PeakLabelThr_Frac)
PeakLabelThr_Frac=1.d-5             !maximum deviation from theory for assigning peak labels (as a fraction of the fragment m/z, used only if this is > PeakLabelThr_Abs)


open(1,file='LABEL_INPUT')
read(nml=INPUT,unit=1,iostat=ierr)
if(ierr .ne. 0) then
 write(*,*) 'stop: failed to read LABEL_INPUT file'
 STOP
end if


open(unit=2,file='mz_data',status='old')
read(2,*) nFragments
allocate(mz_arr(nFragments))
do fragment=1, nFragments
  read(2,*) mz_arr(fragment)
end do
close(2)

MaxLabels=10

allocate(PeakDiff(nFragments),PeakLabel(nFragments),Frag_INFO(nFragments,MaxLabels,4))
PeakLabel(:)='No Match'
PeakDiff(:)=-1000
Frag_INFO(:,:,:)=0


peptide_len=len_trim(peptide_name)

allocate(peptide_amino_arr(peptide_len)) !indexes the specific amino acid residue at this position in the peptide
allocate(peptide_mass_arr(peptide_len))  !gives the mass of the amino acid residue at this position (with adjustment for C terminal)


All_Amino_Types=21 !total number of amino acids implemented

!loop over elements in name, get associated mass, put into peptide_mass_arr
do i = 1, peptide_len
  amino=peptide_name(i:i) !set this
  select case(amino)
  case('G')
    peptide_mass_arr(i)=57.02146372d0
    peptide_amino_arr(i)=1
  case('A')
    peptide_mass_arr(i)=71.03711378d0
    peptide_amino_arr(i)=2
  case('S')
    peptide_mass_arr(i)=87.03202840d0 
    peptide_amino_arr(i)=3
  case('P')
    peptide_mass_arr(i)=97.05276385d0 
    peptide_amino_arr(i)=4
  case('V')
    peptide_mass_arr(i)=99.06841391d0 
    peptide_amino_arr(i)=5
  case('T')
    peptide_mass_arr(i)=101.04767847d0 
    peptide_amino_arr(i)=6
  case('C')
    peptide_mass_arr(i)=103.00918478d0 
    peptide_amino_arr(i)=7
  case('I')
    peptide_mass_arr(i)=113.08406398d0 
    peptide_amino_arr(i)=8
  case('L')
    peptide_mass_arr(i)=113.08406398d0 
    peptide_amino_arr(i)=9
  case('N')
    peptide_mass_arr(i)=114.04292744d0 
    peptide_amino_arr(i)=10
  case('D')
    peptide_mass_arr(i)=115.02694302d0 
    peptide_amino_arr(i)=11
  case('Q')
    peptide_mass_arr(i)=128.05857751d0 
    peptide_amino_arr(i)=12
  case('K')
    peptide_mass_arr(i)=128.09496301d0 
    peptide_amino_arr(i)=13
  case('E')
    peptide_mass_arr(i)=129.04259309d0 
    peptide_amino_arr(i)=14
  case('M')
    peptide_mass_arr(i)=131.04048491d0 
    peptide_amino_arr(i)=15
  case('H')
    peptide_mass_arr(i)=137.05891186d0 
    peptide_amino_arr(i)=16
  case('F')
    peptide_mass_arr(i)=147.06841391d0 
    peptide_amino_arr(i)=17
  case('R')
    peptide_mass_arr(i)=156.10111102d0 
    peptide_amino_arr(i)=18
  case('Y')
    peptide_mass_arr(i)=163.06332853d0 
    peptide_amino_arr(i)=19
  case('W')
    peptide_mass_arr(i)=186.07931295d0 
    peptide_amino_arr(i)=20
  case('p')
    peptide_mass_arr(i)= 243.0296d0
    peptide_amino_arr(i)=21
  case default
    write(*,*) "ERROR: amino acid label", amino, "not defined at position", i
    stop
  end select !NOTE: changing the labelling order in peptide_amino_arr will break the residue-specific losses part
end do


!peptide_mass_arr(1)=peptide_mass_arr(1)!+Neutral_H ####why is this wrong???
peptide_mass_arr(peptide_len)=peptide_mass_arr(peptide_len)+Neutral_CONH2


All_PTM_types=2 !total number of PTMs implemented

if(PTM.ne.'0') then
if(len_trim(PTM).ne.peptide_len) then
  write(*,*) 'ERROR: PTMs incorrectly specified'
  STOP
else
allocate(peptide_PTM_arr(peptide_len))
do i = 1, peptide_len
  modification=PTM(i:i) !set this
  select case(modification)
  case('0') !no PTM in this position
    peptide_PTM_arr(i)=0
  case('N') !OH -> NH2 (e.g. at terminal fragment)
    peptide_PTM_arr(i)=1
    peptide_mass_arr(i)=peptide_mass_arr(i)-0.98401558d0
  case default
    write(*,*) "ERROR: PTM label", modification, "not defined at position", i
    stop
  end select !NOTE: changing the labelling order in peptide_amino_arr will break the residue-specific losses part
end do
end if
end if

!note: currently, I only allow one PTM type per position --- in principle, could have more...
!for some PTMs, can also add neutral losses which include that PTM iff peptide_PTM_arr(i) has a certain value (but would need to
!slightly rework how the neutral loss code works)


parent_mass_max=sum(peptide_mass_arr(:))+maxC13


NeuLossTypes=10 !general neutral losses
NeuLossTypes_extra=6 !residue-specific neutral losses

allocate(NeuLossLabel(NeuLossTypes+NeuLossTypes_extra))
allocate(Neumz(NeuLossTypes+NeuLossTypes_extra))
allocate(NeuLossInclude(All_Amino_Types,NeuLossTypes+NeuLossTypes_extra)) !for each residue, include all .true. losses in the set of possible losses
NeuLossInclude(:,1:NeuLossTypes)=.true.
NeuLossInclude(:,NeuLossTypes+1:NeuLossTypes+NeuLossTypes_extra)=.false.
allocate(NeuLossInclude_Combined(NeuLossTypes+NeuLossTypes_extra))

NeuLossLabel(1)=' '
Neumz(1)=0
NeuLossLabel(2)='-H2O'
Neumz(2)=Neutral_H2O
NeuLossLabel(3)='-CO'
Neumz(3)=Neutral_CO
NeuLossLabel(4)='-NH3'
Neumz(4)=Neutral_NH3
NeuLossLabel(5)='-H2O-H2O'
Neumz(5)=2*Neutral_H2O
NeuLossLabel(6)='-H2O-CO'
Neumz(6)=Neutral_H2O+Neutral_CO
NeuLossLabel(7)='-H2O-NH3'
Neumz(7)=Neutral_H2O+Neutral_NH3
NeuLossLabel(8)='-CO-CO'
Neumz(8)=2*Neutral_CO
NeuLossLabel(9)='-CO-NH3'
Neumz(9)=Neutral_CO+Neutral_NH3
NeuLossLabel(10)='-NH3-NH3'
Neumz(10)=2*Neutral_NH3



start_counter=NeuLossTypes !maybe not the neatest way to do this, but will allow me to add/change included neutral losses in future without manually changing all the indices
counter=start_counter
counter=counter+1
NeuLossLabel(counter)='-HCOH'
Neumz(counter)=30.010565d0
NeuLossInclude(3,start_counter+1:counter)=.true. !S
NeuLossInclude(6,start_counter+1:counter)=.true. !T
NeuLossInclude(11,start_counter+1:counter)=.true. !D
NeuLossInclude(12,start_counter+1:counter)=.true. !Q
NeuLossInclude(14,start_counter+1:counter)=.true. !E
NeuLossInclude(17,start_counter+1:counter)=.true. !F

start_counter=counter
counter=counter+1
NeuLossLabel(counter)='-HCOOH'
Neumz(counter)=46.005480d0
NeuLossInclude(11,start_counter+1:counter)=.true. !D
NeuLossInclude(14,start_counter+1:counter)=.true. !E

start_counter=counter
counter=counter+1
NeuLossLabel(counter)='-HNCNH'
Neumz(counter)=42.021798d0
NeuLossInclude(18,start_counter+1:counter)=.true. !R

start_counter=counter
counter=counter+1
NeuLossLabel(counter)='-CH3CH2SCH3'
Neumz(counter)=76.034671d0
counter=counter+1
NeuLossLabel(counter)='-CH3SH'
Neumz(counter)=48.003371d0
counter=counter+1
NeuLossLabel(counter)='-CH2S'
Neumz(counter)=45.987721d0
NeuLossInclude(15,start_counter+1:counter)=.true. !M


if(counter.ne.NeuLossTypes+NeuLossTypes_extra) then
  write(*,*) 'Error in generating neutral loss list'
  stop
end if


min_internal=1 !#### do properly

maxind=NeuLossTypes+NeuLossTypes_extra+1+2*(peptide_len-1)*(NeuLossTypes+NeuLossTypes_extra)*(parent_charge-1)
if(DoInternalFrag) then
  maxind=maxind+(peptide_len-min_internal-1)*0.5*(peptide_len-min_internal)*(NeuLossTypes+NeuLossTypes_extra)*(parent_charge-1)
end if
maxind=maxind*(maxC13+1)
allocate(Fragmz(maxind)) !maxind is an upper bound --- in practice, not all neutral losses will be possible for all fragments (if memory usage in this section ever becomes a problem, this can be optimised)
allocate(FragLabel(maxind))
FragLabel(:)='FRAGMENT LABEL MISSING'
Fragmz(:)=0
allocate(Frag_SFC(maxind,3)) !start, finish, charge
Frag_SFC(:,:)=-1


ind=0


!fragment_max_charge_fact=1.5 !#### linearly reduce the max allowed charge on a fragment (proportional to frag mass) if it has less mass than (parent mass)/(this factor) 
                              !--- does not currently work, as is too prone to deleting real/plausible assignments


!build parent
Fragmz_base=sum(peptide_mass_arr(:))
FragLabel_base='Parent'

do numC13 = 0, maxC13

NeuLossInclude_Combined(:)=.false.
do amino_ind=1, peptide_len !combine the set of all possible neutral losses across all residues
  NeuLossInclude_Combined=NeuLossInclude_Combined.or.NeuLossInclude(peptide_amino_arr(amino_ind),:)
end do
do NeuLossInd = 1, NeuLossTypes+NeuLossTypes_extra !includes no loss
if(NeuLossInclude_Combined(NeuLossInd)) then !if this loss is possible for any residue in this fragment
  charge=parent_charge
  ind=ind+1
  Fragmz(ind)=(Fragmz_base+numC13*C13-Neumz(NeuLossInd)+charge*proton)/charge
  Frag_SFC(ind,1)=1
  Frag_SFC(ind,2)=peptide_len
  Frag_SFC(ind,3)=charge
  write(FragLabel(ind),'(a,a,a,i0,a)') trim(FragLabel_base), trim(NeuLossLabel(NeuLossInd)), ' (+', charge, ')'
  if(numC13.gt.0) then
    write(numC13_char,'(i0)') numC13
    write(FragLabel(ind),'(a,a,a,a)') trim(FragLabel(ind)), ' (', trim(numC13_char) ,'C13)'
  end if
  if((NeuLossInd.eq.1).and.(numC13.eq.0)) then
    write(*,'(f20.8,a,a)') Fragmz(ind),' - ',FragLabel(ind)
  end if
end if
end do

end do !C13



!build b terminal fragments
do frag_length=1, peptide_len-1
  Fragmz_base=sum(peptide_mass_arr(1:frag_length))
  write(FragLabel_base,'(a,i0)') 'b', frag_length

  maxC13_here=min(maxC13,ceiling(1.2*maxC13*Fragmz_base/parent_mass_max)) !scale maxC13 down based on fragment length, erring slightly on the side of allowing more C13
  maxcharge_here=parent_charge !max(1,min(parent_charge,floor(fragment_max_charge_fact*parent_charge*Fragmz_base/parent_mass_max)))

  do numC13 = 0, maxC13_here
 
    NeuLossInclude_Combined(:)=.false.
    do amino_ind=1, frag_length !combine the set of all possible neutral losses across all residues
      NeuLossInclude_Combined=NeuLossInclude_Combined.or.NeuLossInclude(peptide_amino_arr(amino_ind),:)
    end do
    do NeuLossInd = 1, NeuLossTypes+NeuLossTypes_extra !includes no loss
    if(NeuLossInclude_Combined(NeuLossInd)) then !if this loss is possible for any residue in this fragment
      do charge=1, maxcharge_here
        ind=ind+1
        Fragmz(ind)=(Fragmz_base+numC13*C13-Neumz(NeuLossInd)+charge*proton)/charge
        Frag_SFC(ind,1)=1
        Frag_SFC(ind,2)=frag_length
        Frag_SFC(ind,3)=charge
        write(FragLabel(ind),'(a,a,a,i0,a)') trim(FragLabel_base), trim(NeuLossLabel(NeuLossInd)), ' (+', charge, ')'
        if(numC13.gt.0) then
          write(numC13_char,'(i0)') numC13
          write(FragLabel(ind),'(a,a,a,a)') trim(FragLabel(ind)), ' (', trim(numC13_char) ,'C13)'
        end if
        if((NeuLossInd.eq.1).and.(numC13.eq.0)) then
          write(*,'(f20.8,a,a)') Fragmz(ind),' - ',FragLabel(ind)
        end if
      end do
    end if
    end do 
  end do
end do

!build y terminal fragments
do frag_length=1, peptide_len-1
  Fragmz_base=sum(peptide_mass_arr(peptide_len-frag_length+1:peptide_len))
  write(FragLabel_base,'(a,i0)') 'y', frag_length

  maxC13_here=min(maxC13,ceiling(1.2*maxC13*Fragmz_base/parent_mass_max)) !scale maxC13 down based on fragment length, erring slightly on the side of allowing more C13
  maxcharge_here=parent_charge !max(1,min(parent_charge,floor(fragment_max_charge_fact*parent_charge*Fragmz_base/parent_mass_max)))

  do numC13 = 0, maxC13_here

    NeuLossInclude_Combined(:)=.false.
    do amino_ind=peptide_len-frag_length+1, peptide_len !combine the set of all possible neutral losses across all residues
      NeuLossInclude_Combined=NeuLossInclude_Combined.or.NeuLossInclude(peptide_amino_arr(amino_ind),:)
    end do
    do NeuLossInd = 1, NeuLossTypes+NeuLossTypes_extra !includes no loss
    if(NeuLossInclude_Combined(NeuLossInd)) then !if this loss is possible for any residue in this fragment
      do charge=1, maxcharge_here
        ind=ind+1
        Fragmz(ind)=(Fragmz_base+numC13*C13-Neumz(NeuLossInd)+charge*proton)/charge
        Frag_SFC(ind,1)=peptide_len-frag_length+1
        Frag_SFC(ind,2)=peptide_len
        Frag_SFC(ind,3)=charge
        write(FragLabel(ind),'(a,a,a,i0,a)') trim(FragLabel_base), trim(NeuLossLabel(NeuLossInd)), ' (+', charge, ')'
        if(numC13.gt.0) then
          write(numC13_char,'(i0)') numC13
          write(FragLabel(ind),'(a,a,a,a)') trim(FragLabel(ind)), ' (', trim(numC13_char) ,'C13)'
        end if
        if((NeuLossInd.eq.1).and.(numC13.eq.0)) then
          write(*,'(f20.8,a,a)') Fragmz(ind),' - ',FragLabel(ind)
        end if
      end do
    end if
    end do 
  end do
end do


!build internal fragments 
if(DoInternalFrag) then
do frag_start=2, peptide_len-min_internal !for now, do only internal fragments of length 2 or more
  do frag_length=min_internal, peptide_len-frag_start
    Fragmz_base=sum(peptide_mass_arr(frag_start:frag_start+frag_length-1))
    write(FragLabel_base,'(a,i0,a,i0)') 'internal ', frag_start, ':', frag_start+frag_length-1

    maxC13_here=min(maxC13,ceiling(1.2*maxC13*Fragmz_base/parent_mass_max)) !scale maxC13 down based on fragment length, erring slightly on the side of allowing more C13
    maxcharge_here_internal=parent_charge-1 !max(1,min(parent_charge-1,floor(fragment_max_charge_fact*parent_charge*Fragmz_base/parent_mass_max-1)))

    do numC13 = 0, maxC13_here

      NeuLossInclude_Combined(:)=.false.
      do amino_ind=frag_start, frag_start+frag_length-1 !combine the set of all possible neutral losses across all residues
        NeuLossInclude_Combined=NeuLossInclude_Combined.or.NeuLossInclude(peptide_amino_arr(amino_ind),:)
      end do
      do NeuLossInd = 1, NeuLossTypes+NeuLossTypes_extra !includes no loss
      if(NeuLossInclude_Combined(NeuLossInd)) then !if this loss is possible for any residue in this fragment
        do charge=1, maxcharge_here_internal
          ind=ind+1
          Fragmz(ind)=(Fragmz_base+numC13*C13-Neumz(NeuLossInd)+charge*proton)/charge
          Frag_SFC(ind,1)=frag_start
          Frag_SFC(ind,2)=frag_start+frag_length-1
          Frag_SFC(ind,3)=charge
          write(FragLabel(ind),'(a,a,a,i0,a)') trim(FragLabel_base), trim(NeuLossLabel(NeuLossInd)), ' (+', charge, ')'
          if(numC13.gt.0) then
            write(numC13_char,'(i0)') numC13
            write(FragLabel(ind),'(a,a,a,a)') trim(FragLabel(ind)), ' (', trim(numC13_char) ,'C13)'
          end if
        end do
      end if
      end do 
    end do
  end do
end do
end if


Total_To_Match=ind


!assign peaks
!nfragments=size(mz_arr)
do fragment=1, nfragments
  do ind=1,Total_To_Match
    if((abs(Fragmz(ind)-mz_arr(fragment)).lt.PeakLabelThr_Abs).or.(abs(Fragmz(ind)-mz_arr(fragment)).lt.PeakLabelThr_Frac*Fragmz(ind))) then
      if(Frag_INFO(fragment,1,4).eq.0) then
        PeakLabel(fragment)=FragLabel(ind)
        PeakDiff(fragment)=(Fragmz(ind)-mz_arr(fragment))
        Frag_INFO(fragment,1,1:3)=Frag_SFC(ind,1:3)
        Frag_INFO(fragment,1,4)=1
      else
        write(*,*) 'Warning: multiple matching fragments for peak', fragment, 'mz', mz_arr(fragment)
        Frag_INFO(fragment,1,4)=Frag_INFO(fragment,1,4)+1
        label_num=Frag_INFO(fragment,1,4)
        if(label_num.le.size(Frag_INFO,DIM=2)) then
          Frag_INFO(fragment,label_num,1:3)=Frag_SFC(ind,1:3)
        else
          write(*,*) 'WARNING: could not write fragment info for fragment ', fragment, ': not enough space'
        end if
        if(len(trim(PeakLabel(Fragment)))+len(trim(FragLabel(ind)))+3 .le. len(PeakLabel(Fragment))) then
          PeakLabelTemp=PeakLabel(fragment)
          write(PeakLabel(fragment),'(a,a,a)') trim(PeakLabelTemp), ' / ', trim(FragLabel(ind))
        else
          write(*,*) 'WARNING: could not write fragment label for fragment ', fragment, ': not enough space'
        end if         
      end if
    end if
  end do
end do


open(unit=2,file='FragmentLabels')
do fragment=1, nFragments
  write(2,'(2f20.8,2a)') mz_arr(fragment),PeakDiff(fragment),'   ', trim(PeakLabel(fragment))
end do
close(2)





end program assign_peaks
