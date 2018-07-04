! module to store feature info

module feature_config

    !==============================!
    !* data structure definitions *!
    !==============================!

    !-------------------------------!
    !* types for distances, angles *!
    !-------------------------------!

    type,public :: feature_info_twobody
        integer :: n                        ! number of 2-body terms
        real(8),allocatable :: dr(:)        ! distance to neighbour
        real(8),allocatable :: dr_taper(:)  ! tapering applied to dr
        real(8),allocatable :: dr_taper_deriv(:)
        integer,allocatable :: idx(:)       ! index of neighbour
        real(8),allocatable :: drdri(:,:)   ! derivative of distance wrt atom
        real(8),allocatable :: z(:)         ! atomic number of neighbour
        real(8) :: z_atom                   ! central atomic number
        real(8) :: r_nl_atom(1:3)           ! local position of central atom
        real(8),allocatable :: r_nl_neigh(:,:) ! non-local position of neighburs
    end type feature_info_twobody

    type,public :: feature_info_threebody
        integer :: n                            ! number of 3-body terms
        real(8),allocatable :: cos_ang(:)       ! dtheta_{ijk}
        real(8),allocatable :: dr(:,:)          ! displacements between atoms
        real(8),allocatable :: dr_taper(:,:)    ! tapering applied to dr
        real(8),allocatable :: dr_taper_deriv(:,:)
        real(8),allocatable :: z(:,:)           ! atomic number of neighbours
        integer,allocatable :: idx(:,:)         ! identifier of neighbours
        real(8) :: z_atom                       ! atomic number of central atom
        real(8),allocatable :: dcos_dr(:,:,:)   ! derivative of cos(angle) wrt atoms
        real(8),allocatable :: drdri(:,:,:)     ! derivative of atom-atom displacements
        real(8),allocatable :: r_nl(:,:,:)      ! non-local position of atoms in bond
    end type feature_info_threebody

    !------------------------------!
    !* types for feature metadata *!
    !------------------------------!
    
    type,public :: feature_
        !=======================================================!
        ! ftype : 0 = atomic number                             !
        !         1 = Behler isotorpic                          !
        !         2 = Behler anisotropic                        !
        !         3 = normal isotorpic                          !
        !         4 = normal anisotropic                        !
        !=======================================================!
        integer :: ftype                    ! feature type
        real(8) :: rcut     = 0.0d0         ! interaction cut off 
        real(8) :: rs       = 0.0d0         ! iso exp offset
        real(8) :: fs       = 0.0d0         ! tapering smoothness
        real(8) :: eta      = 0.0d0         ! iso
        real(8) :: xi       = 0.0d0         ! ani
        real(8) :: lambda   = 0.0d0         ! ani
        real(8) :: sqrt_det = 0.0d0         ! precision matrix determinant
        real(8), allocatable :: prec(:,:)   ! normal precision
        real(8), allocatable :: mean(:)     ! normal mean
        real(8) :: za       = 0.0d0         ! central atomic number
        real(8) :: zb       = 0.0d0         ! neighbour atomic number
        real(8) :: scl_cnst = 1.0d0         ! scale all instances of this feature upon computation
        real(8) :: add_cnst = 0.0d0         ! additive constant for this feature upon computation
        real(8), allocatable :: devel(:)    ! for developing functions
        real(8), allocatable :: linear_w(:)
        real(8) :: z_single_element = 0.0d0 ! for when Z contribution can be factorised
        logical :: is_twobody = .false.     ! whether feature is twobody
        logical :: is_threebody = .false.   ! whether feature is threebody
    end type feature_

    type,public :: feature_info
        type(feature_),allocatable,public :: info(:)
        integer :: num_features = 0
        logical :: pca
        real(8) :: pca_threshold
    end type feature_info

    type,public :: neigh_info
        type(feature_info_twobody),allocatable :: twobody(:)
        type(feature_info_threebody),allocatable :: threebody(:)
    end type neigh_info

    !===================!
    !* initialisations *!
    !===================!

    !* feature parameters meta data    
    type(feature_info),save :: feature_params

    !* two body information for a single structure used to generate features
    type(feature_info_twobody),allocatable :: feature_isotropic(:)

    !* three body information for a single structure used to generate features
    type(feature_info_threebody),allocatable :: feature_threebody_info(:)

    !* for storing all neighbourhood info of a single set
    type(neigh_info),allocatable :: set_neigh_info(:)

    !* whether or not to use low mem (slow performance) or high mem (high performance)
    logical :: performance_options(1:6) = .false. 

    !* which physical properties should we calculate
    logical :: computation_options(1:2) = .false.

    !* type of calculation being performaed
    logical :: computation_type(1:3) = .false.

    !* whether nearest image or full periodic boundaries are in use
    logical :: periodic_boundary_convention(1:2) = .false.

    !* openMP pragma necessary for globally scoped variables
    !$omp threadprivate(feature_isotropic)
    !$omp threadprivate(feature_threebody_info)

    contains

        integer function featureID_StringToInt(feature_descriptor)
            !=======================================================!
            ! convert a string identifier for symmetry function to  !
            ! an integer identifier                                 !
            !=======================================================!

            implicit none

            character(len=*),intent(in) :: feature_descriptor
            
            integer :: idx

            idx = -1

            if (feature_descriptor.eq."atomic_number") then
                idx = 0
            else if (feature_descriptor.eq."acsf_behler-g1") then
                idx = 1
            else if (feature_descriptor.eq."acsf_behler-g2") then
                idx = 2
            else if (feature_descriptor.eq."acsf_behler-g4") then
                idx = 3
            else if (feature_descriptor.eq."acsf_behler-g5") then
                idx = 4
            else if (feature_descriptor.eq."acsf_normal-b2") then
                idx = 5
            else if (feature_descriptor.eq."acsf_normal-b3") then
                idx = 6
            else if (feature_descriptor.eq."acsf_fourier-b2") then
                idx = 7
            else if (feature_descriptor.eq."devel_iso") then
                idx = 8
            else
                write(*,*) ""
                write(*,*) '*********************************************' 
                write(*,*) 'error raised in routine : feature_StringToInt'
                write(*,*) '*********************************************' 
                write(*,*) ""
                write(*,*) 'Error : ',"unrecognised feature type ",feature_descriptor
                write(*,*) ""
                call exit(0)
            end if
            featureID_StringToInt = idx
        end function featureID_StringToInt

        logical function feature_IsTwoBody(ftype)
            implicit none

            integer,intent(in) :: ftype
        
            logical :: tmp
            
            if ( (ftype.eq.featureID_StringToInt("acsf_behler-g1")).or.&
                &(ftype.eq.featureID_StringToInt("acsf_behler-g2")).or.&
                &(ftype.eq.featureID_StringToInt("acsf_normal-b2")).or.&
                &(ftype.eq.featureID_StringToInt("acsf_fourier-b2")).or.&
                &(ftype.eq.featureID_StringToInt("devel_iso")) ) then
                tmp = .true.
            else
                tmp = .false.
            end if
            feature_IsTwoBody = tmp
        end function
        
        logical function feature_IsThreeBody(ftype)
            implicit none

            integer,intent(in) :: ftype
        
            logical :: tmp

            if ( (ftype.eq.featureID_StringToInt("acsf_behler-g4")).or.&
                &(ftype.eq.featureID_StringToInt("acsf_behler-g5")).or.&
                &(ftype.eq.featureID_StringToInt("acsf_normal-b3")) ) then
                tmp = .true.
            else
                tmp = .false.
            end if
            feature_IsThreeBody = tmp
        end function feature_IsThreeBody
        
        integer function SpeedUpID_StringToIdx(speedup)
            !=======================================================!
            ! convert a string identifier for symmetry function to  !
            ! an integer identifier                                 !
            !=======================================================!

            implicit none

            character(len=*),intent(in) :: speedup
            
            integer :: idx = -1

            if (speedup.eq."twobody_rcut") then
                !* all two body features have same rcut 
                idx = 1
            else if (speedup.eq."threebody_rcut") then
                !* all threebody features have same rcut
                idx = 2
            else if (speedup.eq."keep_all_neigh_info") then
                !* keep all nearest neighbour info once computed
                idx = 3
            else if (speedup.eq."single_element") then
                !* single element in all train,holdout,test confs
                idx = 4
            else if (speedup.eq."single_element_all_equal") then
                idx = 5
            else if (speedup.eq."lookup_tables") then
                idx = 6
            else
                write(*,*) ""
                write(*,*) "***********************************************"
                write(*,*) "error raised in routine : SpeedUpID_StringToIdx" 
                write(*,*) "***********************************************"
                write(*,*) ""
                write(*,*) "Error : unrecognised speed up",speedup
                write(*,*) ""
                call exit(0)
            end if
    
            if ((idx.lt.1).or.(idx.gt.size(performance_options))) then
                write(*,*) ""
                write(*,*) "***********************************************"
                write(*,*) "error raised in routine : SpeedUpID_StringToIdx" 
                write(*,*) "***********************************************"
                write(*,*) ""
                write(*,*) "Error : performance_options array is wrong shape"
                write(*,*) ""
                call exit(0)
            end if

            SpeedUpID_StringToIdx = idx
        end function SpeedUpID_StringToIdx

        logical function speedup_applies(speedup)
            implicit none

            character(len=*),intent(in) :: speedup

            speedup_applies = performance_options(SpeedUpID_StringToIdx(speedup))
        end function speedup_applies

        subroutine activate_performance_option(speedup)
            implicit none

            character(len=*),intent(in) :: speedup

            performance_options(SpeedUpID_StringToIdx(speedup)) = .true.
        end subroutine activate_performance_option

        logical function calculate_property(property)
            implicit none

            character(len=*),intent(in) :: property

            calculate_property = computation_options(ComputationID_StringToIdx(property))
        end function calculate_property

        integer function ComputationID_StringToIdx(property)
            implicit none

            !* args
            character(len=*),intent(in) :: property

            !* scratch
            integer :: idx=-1

            if (property.eq."stress") then
                idx = 1
            else if (property.eq."forces") then
                idx = 2
            else
                write(*,*) ""
                write(*,*) "***************************************************"
                write(*,*) "error raised in routine : ComputationID_StringToIdx" 
                write(*,*) "***************************************************"
                write(*,*) ""
                write(*,*) "Error : unrecognised property",property
                write(*,*) ""
                call exit(0)
            end if
    
            if ((idx.lt.1).or.(idx.gt.size(computation_options))) then
                write(*,*) ""
                write(*,*) "***************************************************"
                write(*,*) "error raised in routine : ComputationID_StringToIdx" 
                write(*,*) "***************************************************"
                write(*,*) ""
                write(*,*) "Error : computation_options array is wrong shape"
                write(*,*) ""
                call exit(0)
            end if
            ComputationID_StringToIdx = idx
        end function ComputationID_StringToIdx

        subroutine switch_property(property,state)
            implicit none

            character(len=*),intent(in) :: property,state

            logical :: logical_value

            if (state.eq."on") then
                logical_value = .true.
            else if (state.eq."off") then
                logical_value = .false.
            else
                write(*,*) ""
                write(*,*) "*****************************************"
                write(*,*) "error raised in routine : switch_property" 
                write(*,*) "*****************************************"
                write(*,*) ""
                write(*,*) "Error : state",state,"is not recognised"
                write(*,*) ""
                call exit(0)
            end if

            !* turn on or off property
            computation_options(ComputationID_StringToIdx(property)) = logical_value
        end subroutine switch_property
        
        subroutine switch_performance_option(speedup,state)
            implicit none

            character(len=*),intent(in) :: speedup,state

            logical :: logical_value

            if (state.eq."on") then
                logical_value = .true.
            else if (state.eq."off") then
                logical_value = .false.
            else
                write(*,*) ""
                write(*,*) "***************************************************"
                write(*,*) "error raised in routine : switch_performance_option" 
                write(*,*) "***************************************************"
                write(*,*) ""
                write(*,*) "Error : state",state,"is not recognised"
                write(*,*) ""
                call exit(0)
            end if

            !* turn on or off property
            performance_options(SpeedUpID_StringToIdx(speedup)) = logical_value 
        end subroutine switch_performance_option
    
        logical function image_convention(boundary_type)
            implicit none

            character(len=*),intent(in) :: boundary_type

            image_convention = periodic_boundary_convention(BoundaryID_StringToInt(boundary_type))
        end function image_convention

        integer function BoundaryID_StringToInt(boundary_type)
            implicit none

            character(len=*),intent(in) :: boundary_type
            integer :: res

            if (boundary_type.eq."full_periodic") then
                res = 1
            else if (boundary_type.eq."nearest_image") then
                res = 2
            else
                write(*,*) ""
                write(*,*) "*************************************************"
                write(*,*) "error raised in routine : BoundaryID_StringToInt" 
                write(*,*) "************************************************"
                write(*,*) ""
                write(*,*) "Error : image convention",boundary_type,"is not recognised"
                write(*,*) ""
                call exit(0)
            end if
            BoundaryID_StringToInt = res
        end function BoundaryID_StringToInt

        subroutine set_image_convention(boundary_type)
            implicit none

            character(len=*),intent(in) :: boundary_type

            if (boundary_type.eq."full_periodic") then
                periodic_boundary_convention(BoundaryID_StringToInt("full_periodic")) = .true.
                periodic_boundary_convention(BoundaryID_StringToInt("nearest_image")) = .false.
            else if (boundary_type.eq."nearest_image") then
                periodic_boundary_convention(BoundaryID_StringToInt("full_periodic")) = .false.
                periodic_boundary_convention(BoundaryID_StringToInt("nearest_image")) = .true.
            else
                write(*,*) ""
                write(*,*) "**********************************************"
                write(*,*) "error raised in routine : set_image_convention" 
                write(*,*) "**********************************************"
                write(*,*) ""
                write(*,*) "Error : image convention",boundary_type,"is not recognised"
                write(*,*) ""
                call exit(0)
            end if
        end subroutine set_image_convention

        integer function CalculationID_StringToInt(calc_type)
            implicit none

            character(len=*),intent(in) :: calc_type

            integer :: res = -1

            if (calc_type.eq."single_point") then
                res = 1
            else if (calc_type.eq."optimize_net") then
                res = 2
            else if (calc_type.eq."optimize_features") then
                res = 3
            else
                write(*,*) ""
                write(*,*) "***************************************************"
                write(*,*) "error raised in routine : CalculationID_StringToInt" 
                write(*,*) "***************************************************"
                write(*,*) ""
                write(*,*) "Error : calculation type",calc_type,"is not recognised"
                write(*,*) ""
                call exit(0)
            end if
            CalculationID_StringToInt = res
        end function CalculationID_StringToInt

        subroutine set_calculation_type(calc_type)
            implicit none

            character(len=*),intent(in) :: calc_type

            if (calc_type.eq."single_point") then
                computation_type(CalculationID_StringToInt("single_point")) = .true.
                computation_type(CalculationID_StringToInt("optimize_net")) = .false.
                computation_type(CalculationID_StringToInt("optimize_features")) = .false.
            else if (calc_type.eq."optimize_net") then
                computation_type(CalculationID_StringToInt("single_point")) = .false.
                computation_type(CalculationID_StringToInt("optimize_net")) = .true.
                computation_type(CalculationID_StringToInt("optimize_features")) = .false.
            else if (calc_type.eq."optimize_features") then
                computation_type(CalculationID_StringToInt("single_point")) = .false.
                computation_type(CalculationID_StringToInt("optimize_net")) = .false.
                computation_type(CalculationID_StringToInt("optimize_features")) = .true.
            else
                write(*,*) ""
                write(*,*) "**********************************************"
                write(*,*) "error raised in routine : set_calculation_type" 
                write(*,*) "**********************************************"
                write(*,*) ""
                write(*,*) "Error : calculation type",calc_type,"is not recognised"
                write(*,*) ""
                call exit(0)
            end if
        end subroutine set_calculation_type

        logical function calculation_type(calc_type)
            implicit none

            character(len=*),intent(in) :: calc_type

            calculation_type = computation_type(CalculationID_StringToInt(calc_type))
        end function calculation_type
end module feature_config
